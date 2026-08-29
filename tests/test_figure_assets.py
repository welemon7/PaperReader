from PIL import Image, ImageDraw

from src.utils.figure_assets import copy_or_rasterize_asset, crop_content_with_padding, save_svg_asset


def test_crop_content_with_padding_removes_white_margins(tmp_path):
    source_path = tmp_path / "figure.png"
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((30, 20, 69, 59), fill="black")
    image.save(source_path)

    cropped, metadata = crop_content_with_padding(source_path, padding=5)

    assert cropped.size == (50, 50)
    assert metadata == {
        "original_size": [100, 80],
        "cropped_size": [50, 50],
        "aspect_ratio": 1.0,
    }


def test_crop_content_with_padding_returns_original_for_all_white_image(tmp_path):
    source_path = tmp_path / "blank.png"
    Image.new("RGBA", (32, 24), (255, 255, 255, 128)).save(source_path)

    cropped, metadata = crop_content_with_padding(source_path)

    assert cropped.size == (32, 24)
    assert metadata["original_size"] == [32, 24]
    assert metadata["cropped_size"] == [32, 24]
    assert metadata["aspect_ratio"] == 32 / 24


def test_copy_or_rasterize_asset_crops_copied_image(tmp_path):
    source_path = tmp_path / "figure.png"
    image = Image.new("RGB", (60, 40), "white")
    ImageDraw.Draw(image).rectangle((20, 10, 39, 29), fill="black")
    image.save(source_path)

    target = copy_or_rasterize_asset(source_path, tmp_path / "out", "figure")

    assert target is not None
    with Image.open(target) as prepared:
        assert prepared.size == (40, 40)


def test_copy_or_rasterize_asset_preserves_svg_assets(tmp_path):
    source_path = tmp_path / "vector.svg"
    source_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M1 1h8v8H1z"/></svg>',
        encoding="utf-8",
    )

    target = copy_or_rasterize_asset(source_path, tmp_path / "out", "vector")

    assert target is not None
    assert target.suffix == ".svg"
    assert target.exists()
    assert "<svg" in target.read_text(encoding="utf-8")


def test_save_svg_asset_wraps_symbol_text(tmp_path):
    target = save_svg_asset("◎", tmp_path / "figures", "symbol-highlight")

    assert target.exists()
    assert target.suffix == ".svg"
    content = target.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "◎" in content


def test_save_svg_asset_preserves_prefixed_svg_document(tmp_path):
    source = '<ns0:svg xmlns:ns0="http://www.w3.org/2000/svg"><ns0:rect width="10" height="10"/></ns0:svg>'
    target = save_svg_asset(source, tmp_path / "figures", "prefixed")
    content = target.read_text(encoding="utf-8")
    assert "&lt;ns0:svg" not in content
    assert "<ns0:svg" in content
