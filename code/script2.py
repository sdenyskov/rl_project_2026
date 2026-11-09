from PIL import Image
import math

for code in ["beta00", "beta05", "beta10"]:

    images = [
        Image.open(f"recordings/screenshot_{code}_step{step}.png")
        for step in range(0, 500, 50)
    ]

    # 5 columns × 2 rows
    cols = 5
    rows = 2

    # Use the largest common size that fits all images without distortion
    width = min(img.width for img in images)
    height = min(img.height for img in images)

    # Resize while preserving aspect ratio
    resized = []
    for img in images:
        scale = min(width / img.width, height / img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        resized.append(img.resize(new_size, Image.Resampling.LANCZOS))

    # Determine cell size from largest resized dimensions
    cell_width = max(img.width for img in resized)
    cell_height = max(img.height for img in resized)

    # Create grid
    grid = Image.new(
        "RGB",
        (cols * cell_width, rows * cell_height),
        "white"
    )

    for i, img in enumerate(resized):
        x = (i % cols) * cell_width
        y = (i // cols) * cell_height

        # Centre image within its cell
        x += (cell_width - img.width) // 2
        y += (cell_height - img.height) // 2

        grid.paste(img, (x, y))

    grid.save(f"recordings/simulation_grid_{code}.png")
