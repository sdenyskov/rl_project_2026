from PIL import Image

X = 25
code = "TEST00_TEST00"

gif = Image.open(f"media/gifs/first10x10run_6-test-iters30_{code}.gif")
for step in range(0, gif.n_frames, X):
    if step == 0:
        gif.seek(1)
    else:
        gif.seek(step)
    frame = gif.convert("RGB")
    frame.save(f"recordings/screenshot_beta00_step{step}.png")

gif = Image.open(f"media/gifs/first10x10run_65-test-iters30_{code}.gif")
for step in range(0, gif.n_frames, X):
    if step == 0:
        gif.seek(1)
    else:
        gif.seek(step)
    frame = gif.convert("RGB")
    frame.save(f"recordings/screenshot_beta05_step{step}.png")

gif = Image.open(f"media/gifs/first10x10run_7-test-iters30_{code}.gif")
for step in range(0, gif.n_frames, X):
    if step == 0:
        gif.seek(1)
    else:
        gif.seek(step)
    frame = gif.convert("RGB")
    frame.save(f"recordings/screenshot_beta10_step{step}.png")
