from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings
from src.schemas.paper import PaperDocument

logger = logging.getLogger(__name__)


# ---- ORM models ----

class Base(DeclarativeBase):
    pass


class PaperRecord(Base):
    __tablename__ = "papers"

    paper_id = Column(String, primary_key=True)
    arxiv_id = Column(String, unique=True, nullable=False)
    document_json = Column(Text, nullable=False)  # full PaperDocument as JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class SectionRecord(Base):
    __tablename__ = "sections"

    section_id = Column(String, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.paper_id"), nullable=False, primary_key=True)
    parent_id = Column(String, nullable=True)
    title = Column(String, nullable=False)
    level = Column(Integer, nullable=False)
    text = Column(Text, default="")
    raw_latex = Column(Text, default="")


class FormulaRecord(Base):
    __tablename__ = "formulas"

    formula_id = Column(String, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.paper_id"), nullable=False, primary_key=True)
    section_id = Column(String, nullable=False)
    latex = Column(Text, nullable=False)
    semantic_desc = Column(Text, default="")
    label = Column(String, nullable=True)


class FigureRecord(Base):
    __tablename__ = "figures"

    figure_id = Column(String, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.paper_id"), nullable=False, primary_key=True)
    section_id = Column(String, nullable=False)
    caption = Column(Text, default="")
    asset_filename = Column(String, nullable=True)
    minio_path = Column(String, nullable=True)
    local_path = Column(String, nullable=True)
    label = Column(String, nullable=True)
    width = Column(String, nullable=True)


class ReferenceRecord(Base):
    __tablename__ = "references"

    ref_id = Column(String, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.paper_id"), nullable=False, primary_key=True)
    bibkey = Column(String, nullable=False)
    title = Column(Text, default="")
    authors = Column(String, default="")
    journal = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    doi = Column(String, nullable=True)


# ---- Database class ----

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    paper_id = Column(String, primary_key=True)
    arxiv_id = Column(String, nullable=False)
    analysis_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PaperDatabase:
    """SQLite storage for parsed paper data."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        db_url = db_url or settings.database_url
        # Ensure data directory exists for file-based SQLite
        if db_url.startswith("sqlite"):
            path_part = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
            Path(path_part).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            db_url.replace("+aiosqlite", "+pysqlite"),  # use sync driver for management
            echo=False,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("Database initialized: %s", db_url)

    def save_paper(self, doc: PaperDocument) -> str:
        """Save the full PaperDocument into SQLite.
        Returns paper_id.
        """
        paper_id = doc.paper_id
        with self.Session() as session:
            # Upsert: delete existing records first
            self._delete_paper(session, paper_id)

            # Paper record
            paper_rec = PaperRecord(
                paper_id=paper_id,
                arxiv_id=doc.arxiv_id,
                document_json=doc.model_dump_json(indent=2),
                created_at=datetime.utcnow(),
            )
            session.add(paper_rec)

            # Sections
            for sec in doc.sections:
                self._save_section(session, paper_id, sec, parent_id=None)

            # Formulas
            for f in doc.formulas:
                session.add(
                    FormulaRecord(
                        formula_id=f.formula_id,
                        paper_id=paper_id,
                        section_id=f.section_id,
                        latex=f.latex,
                        semantic_desc=f.semantic_desc,
                        label=f.label,
                    )
                )

            # Figures
            for fig in doc.figures:
                session.add(
                    FigureRecord(
                        figure_id=fig.figure_id,
                        paper_id=paper_id,
                        section_id=fig.section_id,
                        caption=fig.caption,
                        asset_filename=fig.asset_filename,
                        minio_path=fig.minio_path,
                        local_path=fig.local_path,
                        label=fig.label,
                        width=fig.width,
                    )
                )

            # References
            for ref in doc.references:
                session.add(
                    ReferenceRecord(
                        ref_id=ref.ref_id,
                        paper_id=paper_id,
                        bibkey=ref.bibkey,
                        title=ref.title,
                        authors=ref.authors,
                        journal=ref.journal,
                        year=ref.year,
                        doi=ref.doi,
                    )
                )

            session.commit()
            logger.info("Saved paper %s (%s) to database", paper_id, doc.arxiv_id)

        return paper_id

    def get_paper(self, paper_id: str) -> Optional[PaperDocument]:
        """Retrieve a PaperDocument by paper_id."""
        with self.Session() as session:
            rec = session.get(PaperRecord, paper_id)
            if not rec:
                return None
            return PaperDocument.model_validate_json(rec.document_json)

    def get_paper_by_arxiv(self, arxiv_id: str) -> Optional[PaperDocument]:
        """Retrieve a PaperDocument by arXiv ID."""
        with self.Session() as session:
            stmt = select(PaperRecord).where(PaperRecord.arxiv_id == arxiv_id)
            rec = session.execute(stmt).scalar_one_or_none()
            if not rec:
                return None
            return PaperDocument.model_validate_json(rec.document_json)

    def list_papers(self) -> list[dict]:
        """Return a summary list of stored papers."""
        with self.Session() as session:
            rows = session.execute(
                select(
                    PaperRecord.paper_id,
                    PaperRecord.arxiv_id,
                    PaperRecord.created_at,
                ).order_by(PaperRecord.created_at.desc())
            ).all()
            return [
                {"paper_id": r[0], "arxiv_id": r[1], "created_at": r[2].isoformat() if r[2] else ""}
                for r in rows
            ]

    # ---- helpers ----

    @staticmethod
    def _delete_paper(session: Session, paper_id: str) -> None:
        """Remove all records for a paper (for upsert)."""
        for table in (ReferenceRecord, FigureRecord, FormulaRecord, SectionRecord, PaperRecord):
            session.query(table).filter(table.paper_id == paper_id).delete()
        session.flush()

    @staticmethod
    def _save_section(
        session: Session,
        paper_id: str,
        sec,
        parent_id: Optional[str],
    ) -> None:
        """Recursively save a section and its children."""
        rec = SectionRecord(
            section_id=sec.section_id,
            paper_id=paper_id,
            parent_id=parent_id,
            title=sec.title,
            level=sec.level,
            text=sec.text,
            raw_latex=sec.raw_latex,
        )
        session.add(rec)
        for sub in sec.subsections:
            PaperDatabase._save_section(session, paper_id, sub, parent_id=sec.section_id)

    def save_analysis(self, analysis) -> None:
        from src.schemas.analysis import PaperAnalysis
        with self.Session() as session:
            rec = AnalysisRecord(
                paper_id=analysis.paper_id,
                arxiv_id=analysis.arxiv_id,
                analysis_json=analysis.model_dump_json(indent=2),
                created_at=datetime.utcnow(),
            )
            session.merge(rec)
            session.commit()
            logger.info("Saved analysis for %s", analysis.arxiv_id)

    def get_analysis(self, paper_id: str):
        from src.schemas.analysis import PaperAnalysis
        with self.Session() as session:
            rec = session.get(AnalysisRecord, paper_id)
            if not rec:
                return None
            return PaperAnalysis.model_validate_json(rec.analysis_json)

    def get_analysis_by_arxiv(self, arxiv_id: str):
        from src.schemas.analysis import PaperAnalysis
        with self.Session() as session:
            stmt = select(AnalysisRecord).where(AnalysisRecord.arxiv_id == arxiv_id)
            rec = session.execute(stmt).scalar_one_or_none()
            if not rec:
                return None
            return PaperAnalysis.model_validate_json(rec.analysis_json)

    def close(self) -> None:
        self.engine.dispose()
