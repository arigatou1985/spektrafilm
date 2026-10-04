import datetime
import os

import exiv2
import numpy as np
import pytest

from spektrafilm.utils import io as io_module
from spektrafilm.utils.io import ImageMetadata, read_image_metadata, save_image_oiio, write_image_metadata


def _build_source_metadata():
    exif = exiv2.ExifData()

    exif["Exif.Image.Make"] = "Canon"
    exif["Exif.Image.Model"] = "Canon EOS 5D Mark IV"
    exif["Exif.Photo.FocalLength"] = "50/1"
    exif["Exif.Photo.LensModel"] = "Canon EF 50mm f/1.8 STM"

    iptc = exiv2.IptcData()
    iptc["Iptc.Application2.Keywords"] = "dummy"

    xmp = exiv2.XmpData()
    xmp["Xmp.dc.creator"] = "Dummy Creator"

    return ImageMetadata(exif=exif, iptc=iptc, xmp=xmp)


def _read_tags(metadata_field):
    return {datum.key(): str(datum.value()) for datum in metadata_field}


def test_read_metadata_returns_none_for_missing_file(tmp_path):
    assert read_image_metadata(str(tmp_path / "does_not_exist.jpg")) is None


def test_write_metadata_carries_source_tags_and_sets_overrides(tmp_path, monkeypatch):
    fixed_now = datetime.datetime(2025, 6, 15, 12, 30, 45)

    class _MonkeyDatetime(datetime.datetime):
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr(io_module.datetime, "datetime", _MonkeyDatetime)

    source_metadata = _build_source_metadata()

    destination_path = tmp_path / "dst.jpg"

    save_image_oiio(str(destination_path), np.random.rand(12, 16, 3))

    write_image_metadata(str(destination_path), source_metadata)

    result = read_image_metadata(str(destination_path))

    exif = _read_tags(result.exif)
    iptc = _read_tags(result.iptc)
    xmp = _read_tags(result.xmp)

    # Copied tags
    assert exif["Exif.Image.Make"] == "Canon"
    assert exif["Exif.Image.Model"] == "Canon EOS 5D Mark IV"
    assert exif["Exif.Photo.FocalLength"] == "50/1"
    assert exif["Exif.Photo.LensModel"] == "Canon EF 50mm f/1.8 STM"

    assert iptc["Iptc.Application2.Keywords"] == "dummy"

    assert xmp["Xmp.dc.creator"] == "Dummy Creator"

    # Overridden tags
    assert exif["Exif.Image.Orientation"] == "1"
    assert exif["Exif.Image.Software"] == "spektrafilm"
    assert exif["Exif.Image.DateTime"] == "2025:06:15 12:30:45"
    assert exif["Exif.Photo.PixelXDimension"] == "16"
    assert exif["Exif.Photo.PixelYDimension"] == "12"


def test_write_metadata_drops_structural_exif_groups(tmp_path):
    """Source sub-image groups must not become TIFF SubIFDs in the saved file.

    NEF/DNG files carry ``Exif.SubImage*`` groups (embedded previews). exiv2
    materialises them as real TIFF ``SubIFDs`` entries, which makes Adobe Camera
    Raw / Lightroom read the saved TIFF as a raw container and render noise
    instead of the image. Descriptive tags must still be copied over.
    """

    source_metadata = _build_source_metadata()
    source_metadata.exif["Exif.SubImage1.NewSubfileType"] = 1
    source_metadata.exif["Exif.SubImage1.JPEGInterchangeFormat"] = 32

    destination_path = tmp_path / "dst.tif"

    save_image_oiio(str(destination_path), np.random.rand(12, 16, 3))

    write_image_metadata(str(destination_path), source_metadata)

    exif = _read_tags(read_image_metadata(str(destination_path)).exif)

    assert "Exif.Image.SubIFDs" not in exif
    assert "Exif.SubImage1.JPEGInterchangeFormat" not in exif

    # Descriptive metadata is still carried over.
    assert exif["Exif.Image.Make"] == "Canon"
    assert exif["Exif.Photo.LensModel"] == "Canon EF 50mm f/1.8 STM"


def test_write_metadata_drops_dng_raw_format_tags(tmp_path):
    """DNG raw-format tags must not be copied onto a saved RGB image.

    A decoder that finds ``DNGVersion`` / ``ColorMatrix1`` / ``AsShotNeutral``
    on an ordinary TIFF treats it as a raw file and refuses to open it -- macOS
    ImageIO, and therefore Preview, returns nil for its dimensions.
    """

    source_metadata = _build_source_metadata()
    source_metadata.exif["Exif.Image.UniqueCameraModel"] = "Apple iPhone 15 Pro"
    source_metadata.exif["Exif.Image.DNGBackwardVersion"] = "1 4 0 0"

    destination_path = tmp_path / "dst.tif"

    save_image_oiio(str(destination_path), np.random.rand(12, 16, 3))

    write_image_metadata(str(destination_path), source_metadata)

    exif = _read_tags(read_image_metadata(str(destination_path)).exif)

    assert "Exif.Image.UniqueCameraModel" not in exif
    assert "Exif.Image.DNGBackwardVersion" not in exif

    # Descriptive metadata is still carried over.
    assert exif["Exif.Image.Make"] == "Canon"
    assert exif["Exif.Image.Model"] == "Canon EOS 5D Mark IV"


def test_save_without_metadata_has_no_exif(tmp_path):
    destination_path = tmp_path / "plain.jpg"

    save_image_oiio(str(destination_path), np.random.rand(4, 4, 3))

    assert os.path.isfile(destination_path)

    result = read_image_metadata(str(destination_path))

    assert "Exif.Image.Software" not in _read_tags(result.exif)


@pytest.mark.parametrize(
    ("saving_color_space", "saving_cctf_encoding", "expected_colorspace", "expected_iop", "expected_profile_name"),
    [
        ("sRGB", True, "1", "R98", "sRGB"),
        ("Adobe RGB (1998)", True, "65535", "R03", "Adobe RGB (1998)"),
        ("Display P3", True, "65535", None, "Display P3"),
        ("ProPhoto RGB", False, "65535", None, "ProPhoto RGB (linear)"),
        ("sRGB", False, "65535", None, "sRGB (linear)"),
    ],
)
def test_write_metadata_records_saving_color_space(
    tmp_path,
    saving_color_space,
    saving_cctf_encoding,
    expected_colorspace,
    expected_iop,
    expected_profile_name,
):
    destination_path = tmp_path / "tagged.jpg"

    save_image_oiio(str(destination_path), np.random.rand(8, 8, 3))
    write_image_metadata(
        str(destination_path),
        saving_color_space=saving_color_space,
        saving_cctf_encoding=saving_cctf_encoding,
    )

    result = read_image_metadata(str(destination_path))
    exif = _read_tags(result.exif)
    xmp = _read_tags(result.xmp)

    assert exif["Exif.Photo.ColorSpace"] == expected_colorspace
    if expected_iop is None:
        assert "Exif.Iop.InteroperabilityIndex" not in exif
    else:
        assert exif["Exif.Iop.InteroperabilityIndex"] == expected_iop
    assert xmp["Xmp.photoshop.ICCProfile"] == expected_profile_name


@pytest.mark.parametrize(
    ("bit_depth", "expected_format"),
    [
        (8, "uint8"),
        (16, "uint16"),
        (32, "float"),
    ],
)
def test_save_image_oiio_tiff_bit_depths_roundtrip(tmp_path, bit_depth, expected_format):
    import OpenImageIO as oiio

    destination_path = tmp_path / f"out_{bit_depth}.tif"
    image_data = np.random.rand(8, 8, 3).astype(np.float32)

    save_image_oiio(str(destination_path), image_data, bit_depth=bit_depth)

    in_img = oiio.ImageInput.open(str(destination_path))
    try:
        spec = in_img.spec()
        assert str(spec.format) == expected_format
        assert spec.width == 8
        assert spec.height == 8
        assert spec.nchannels == 3
        assert spec.getattribute("Compression") == "zip"
    finally:
        in_img.close()


def test_save_image_oiio_writes_jpeg_without_chroma_subsampling(tmp_path):
    """JPEG must be written 4:4:4 (no chroma subsampling).

    OpenImageIO defaults to 4:2:0, which averages chroma over 2x2 blocks and
    visibly desaturates fine colour detail compared with what the GUI shows.
    ``PIL.JpegImagePlugin.get_sampling`` returns 0 for 4:4:4, 1 for 4:2:2 and
    2 for 4:2:0.
    """

    from PIL import Image
    from PIL.JpegImagePlugin import get_sampling

    destination_path = tmp_path / "out.jpg"

    save_image_oiio(str(destination_path), np.random.rand(16, 16, 3))

    with Image.open(destination_path) as image:
        assert get_sampling(image) == 0


@pytest.mark.parametrize("ext", ["jpg", "png", "tif"])
def test_save_image_oiio_embeds_icc_profile_when_available(tmp_path, monkeypatch, ext):
    import OpenImageIO as oiio
    from PIL import ImageCms

    # A real (small) ICC profile so libpng's validation accepts it.
    real_icc_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    monkeypatch.setattr(
        io_module,
        "_load_icc_profile",
        lambda color_space, cctf_encoding: real_icc_bytes,
    )

    destination_path = tmp_path / f"with_icc.{ext}"
    save_image_oiio(
        str(destination_path),
        np.random.rand(8, 8, 3),
        color_space="Display P3",
        cctf_encoding=True,
    )

    in_img = oiio.ImageInput.open(str(destination_path))
    try:
        embedded = in_img.spec().getattribute("ICCProfile")
    finally:
        in_img.close()

    assert embedded is not None
    assert bytes(embedded) == real_icc_bytes


def test_save_image_oiio_skips_icc_when_profile_missing(tmp_path, monkeypatch):
    import OpenImageIO as oiio

    monkeypatch.setattr(io_module, "_load_icc_profile", lambda color_space, cctf_encoding: None)

    destination_path = tmp_path / "no_icc.jpg"
    save_image_oiio(
        str(destination_path),
        np.random.rand(8, 8, 3),
        color_space="ProPhoto RGB",
        cctf_encoding=True,
    )

    in_img = oiio.ImageInput.open(str(destination_path))
    try:
        embedded = in_img.spec().getattribute("ICCProfile")
    finally:
        in_img.close()

    assert embedded is None


def test_write_metadata_without_source_still_writes_overrides(tmp_path):
    destination_path = tmp_path / "no_source.jpg"

    save_image_oiio(str(destination_path), np.random.rand(4, 4, 3))
    write_image_metadata(str(destination_path), saving_color_space="sRGB")

    result = read_image_metadata(str(destination_path))
    exif = _read_tags(result.exif)

    assert exif["Exif.Image.Software"] == "spektrafilm"
    assert exif["Exif.Photo.ColorSpace"] == "1"
