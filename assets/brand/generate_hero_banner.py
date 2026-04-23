from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1440
HEIGHT = 760
ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "hero-banner.png"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def make_linear_gradient(size: tuple[int, int], top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size)
    pixels = image.load()
    for y in range(height):
        mix = y / max(height - 1, 1)
        color = tuple(
            int(top_color[channel] * (1.0 - mix) + bottom_color[channel] * mix)
            for channel in range(3)
        )
        for x in range(width):
            pixels[x, y] = (*color, 255)
    return image


def add_glow(
    canvas: Image.Image,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: int,
) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=42))
    canvas.alpha_composite(layer)


def draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text(position, text, font=font, fill=fill)


def pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text_fill: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill)
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = left + ((right - left - text_width) // 2)
    text_y = top + ((bottom - top - text_height) // 2) - 1
    draw.text((text_x, text_y), text, font=font, fill=text_fill)


def main() -> None:
    image = make_linear_gradient((WIDTH, HEIGHT), (16, 26, 48), (34, 70, 114))
    add_glow(image, center=(1210, 116), radius=230, color=(92, 200, 255), alpha=70)
    add_glow(image, center=(228, 612), radius=250, color=(255, 155, 94), alpha=58)
    draw = ImageDraw.Draw(image)

    panel_fill = (18, 30, 52, 244)
    panel_edge = (62, 122, 162, 255)
    grid = (90, 116, 150, 42)

    draw.rounded_rectangle((0, 0, WIDTH - 1, HEIGHT - 1), radius=32, outline=(22, 40, 64, 255), width=2)

    for y in range(110, 700, 84):
        draw.line((92, y, 1348, y), fill=grid, width=1)
    for x in range(188, 1310, 184):
        draw.line((x, 70, x, 658), fill=grid, width=1)

    draw.polygon(
        [
            (0, 690),
            (132, 630),
            (308, 594),
            (472, 644),
            (610, 686),
            (760, 710),
            (928, 676),
            (1102, 642),
            (1264, 608),
            (1440, 622),
            (1440, 760),
            (0, 760),
        ],
        fill=(23, 51, 90, 255),
    )

    draw.rounded_rectangle((228, 170, 1048, 468), radius=30, fill=panel_fill, outline=panel_edge, width=2)
    draw.rounded_rectangle((1080, 204, 1360, 432), radius=24, fill=(22, 37, 63, 240), outline=(49, 92, 135), width=2)

    label_font = load_font(r"C:\Windows\Fonts\segoeuib.ttf", 22)
    body_font = load_font(r"C:\Windows\Fonts\segoeui.ttf", 25)
    small_font = load_font(r"C:\Windows\Fonts\segoeuib.ttf", 17)
    title_font = load_font(r"C:\Windows\Fonts\georgiab.ttf", 56)
    panel_title_font = load_font(r"C:\Windows\Fonts\segoeuib.ttf", 18)
    panel_body_font = load_font(r"C:\Windows\Fonts\segoeui.ttf", 18)
    mono_font = load_font(r"C:\Windows\Fonts\consola.ttf", 18)
    footer_font = load_font(r"C:\Windows\Fonts\segoeuib.ttf", 30)

    draw_text(draw, (280, 226), "FLASHATTENTION TILING ANALYZER", label_font, (125, 211, 252))
    draw_text(draw, (280, 312), "Replay Prompt Flash", title_font, (248, 250, 252))
    draw_text(draw, (280, 386), "Attention tiling.", title_font, (248, 250, 252))
    draw_text(
        draw,
        (280, 458),
        "Follow the PFA V3 API path into the host-side tiling_v2 implementation.",
        body_font,
        (211, 227, 255),
    )
    draw_text(
        draw,
        (280, 494),
        "Expand logical groups into physical-core work, JSON, CSV, and Q x KV maps.",
        body_font,
        (211, 227, 255),
    )

    pill(draw, (280, 506, 406, 544), (39, 75, 128), "PFA V3 API", small_font, (220, 234, 255))
    pill(draw, (420, 506, 562, 544), (39, 75, 128), "tiling_v2.cpp", small_font, (220, 234, 255))
    pill(draw, (576, 506, 700, 544), (15, 107, 112), "32 cores", small_font, (220, 234, 255))
    pill(draw, (714, 506, 856, 544), (131, 86, 30), "SVG outputs", small_font, (255, 240, 213))

    draw.line((282, 578, 870, 578), fill=(77, 126, 177), width=2)
    for x, color in ((322, (255, 200, 87)), (492, (97, 212, 207)), (662, (140, 199, 255)), (832, (97, 212, 207))):
        draw.ellipse((x - 10, 568, x + 10, 588), fill=color)
    draw_text(draw, (280, 612), "api entry", body_font, (214, 229, 255))
    draw_text(draw, (431, 612), "host split", body_font, (214, 229, 255))
    draw_text(draw, (605, 612), "core replay", body_font, (214, 229, 255))
    draw_text(draw, (765, 612), "svg evidence", body_font, (214, 229, 255))

    draw_text(draw, (1112, 258), "Per-Core View", panel_title_font, (248, 250, 252))
    draw_text(draw, (1112, 288), "vector / cube lanes", panel_body_font, (144, 183, 230))
    for left, top, right, bottom, fill in (
        (1140, 314, 1174, 400, (38, 72, 111)),
        (1188, 288, 1222, 400, (68, 110, 170)),
        (1236, 332, 1270, 400, (114, 214, 210)),
        (1284, 252, 1318, 400, (255, 200, 87)),
    ):
        draw.rounded_rectangle((left, top, right, bottom), radius=10, fill=fill)
    draw.line((1140, 412, 1318, 412), fill=(59, 98, 143), width=2)
    draw_text(draw, (1140, 444), "AIC workload", mono_font, (167, 196, 230))

    draw_text(
        draw,
        (230, 680),
        "Turn testcase uncertainty into source-backed evidence.",
        footer_font,
        (234, 242, 255),
    )

    image.save(OUTPUT_PATH, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
