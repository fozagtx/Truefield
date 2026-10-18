"""Generate samples/closeout_form.pdf + closeout_photo.jpg (printed close-out form) and a voice-note transcript.
Record samples/voice_note.m4a yourself (read voice_note_transcript.txt aloud) for the live Evolus run-media demo."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"
FORM = [
    "TORQUE CLOSE-OUT FORM  (PUMP-07 / BRG-1)",
    "",
    "WO: (as issued)",
    "Technician: Kwame Asante",
    "Root cause: Outer-race spalling on BRG-1",
    "Matches diagnosis: yes",
    "Actions done: Applied LOTO per LOTO-PUMP-07; Replaced bearing BRG-ZA2115; Flushed and refilled oil",
    "Parts used: BRG-ZA2115 x1; OIL-ISO68-5L x2",
    "Actual downtime minutes: 95",
    "LOTO removed by: Kwame Asante",
    "Follow up: Re-check vibration after 24 h",
]
TRANSCRIPT = ("Close-out for the PUMP-07 job. Technician Kwame Asante. Replaced the drive-end bearing, outer race spalling confirmed, "
              "matches the diagnosis. Used one BRG-ZA2115 and two OIL-ISO68-5L. Downtime was 95 minutes. "
              "Locks removed by Kwame Asante. Follow up: re-check vibration after 24 hours.")


def main():
    OUT.mkdir(exist_ok=True)
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    y = 120
    for line in FORM:
        while len(line) > 70:
            cut = line[:70].rfind(" ")
            d.text((90, y), line[:cut], fill="black", font=font); y += 50; line = "    " + line[cut + 1:]
        d.text((90, y), line, fill="black", font=font); y += 60
    img.save(OUT / "closeout_form.pdf", "PDF", resolution=150)
    img.save(OUT / "closeout_photo.jpg", "JPEG", quality=85)
    (OUT / "closeout_form.txt").write_text("\n".join(FORM) + "\n")
    (OUT / "voice_note_transcript.txt").write_text(TRANSCRIPT + "\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
