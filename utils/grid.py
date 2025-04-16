import os
import math
from PIL import Image, ImageDraw, ImageFont


def create_grid_image(
    images,
    prompts,
    category,
    rows=None,
    output_path=None,
    title_size=100,
    prompt_size=80,
    title_font_size=36,
    prompt_font_size=18,
):
    """
    Create a grid image from multiple images with their prompts.

    Args:
        images: List of PIL Image objects
        prompts: List of prompt strings used to generate the images
        category: Category name to display at the top
        rows: Number of rows in the grid (if None, will be calculated)
        output_path: Path to save the grid image (if None, will return the image)
        title_size: Height of title area in pixels
        prompt_size: Height of prompt area in pixels
        title_font_size: Font size for the category title
        prompt_font_size: Font size for the prompts

    Returns:
        PIL Image object if output_path is None, otherwise None
    """
    # Calculate rows and columns
    num_images = len(images)
    if rows is None:
        rows = math.isqrt(num_images)  # Get integer square root
    cols = math.ceil(num_images / rows)

    # Ensure all images are the same size
    width, height = images[0].size

    # Use TTF font from assets folder
    font_path = os.path.join("assets", "NotoSans-Regular.ttf")
    try:
        title_font = ImageFont.truetype(font_path, title_font_size)
        prompt_font = ImageFont.truetype(font_path, prompt_font_size)
    except Exception as e:
        print(f"Error loading font from {font_path}: {e}")
        title_font = ImageFont.load_default()
        prompt_font = ImageFont.load_default()

    # Calculate sizes with configurable areas for better visibility
    title_height = title_size
    prompt_height = prompt_size
    grid_width = cols * width
    grid_height = rows * (height + prompt_height) + title_height

    # Create a blank canvas
    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid_image)

    # Draw title
    draw.rectangle([(0, 0), (grid_width, title_height)], fill=(240, 240, 240))

    # Draw title text with custom font
    try:
        draw.text(
            (grid_width // 2, title_height // 2),
            category,
            fill=(0, 0, 0),
            font=title_font,
            anchor="mm",
        )
    except Exception:
        # Fallback if the font doesn't support anchor
        text_width = (
            title_font.getlength(category)
            if hasattr(title_font, "getlength")
            else len(category) * 15
        )
        draw.text(
            (grid_width // 2 - text_width // 2, title_height // 2 - 10),
            category,
            fill=(0, 0, 0),
            font=title_font,
        )

    # Place images and prompts
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        # Calculate position
        x = col * width
        y = row * (height + prompt_height) + title_height

        # Paste image
        grid_image.paste(img, (x, y))

        # Draw prompt text
        text_y = y + height
        prompt_text = prompt
        if len(prompt_text) > 50:  # Truncate long prompts
            prompt_text = prompt_text[:47] + "..."

        # Draw text background with larger area
        draw.rectangle(
            [(x, text_y), (x + width, text_y + prompt_height)], fill=(240, 240, 240)
        )

        # Draw text with better positioning and custom font
        text_x = x + width // 2
        text_y = y + height + prompt_height // 2

        # Use anchor="mm" for better text centering when using custom font
        try:
            draw.text(
                (text_x, text_y),
                prompt_text,
                fill=(0, 0, 0),
                font=prompt_font,
                anchor="mm",
            )
        except Exception:
            # Fallback if the font doesn't support anchor
            text_width = (
                prompt_font.getlength(prompt_text)
                if hasattr(prompt_font, "getlength")
                else len(prompt_text) * 8
            )
            draw.text(
                (text_x - text_width // 2, text_y - 6),
                prompt_text,
                fill=(0, 0, 0),
                font=prompt_font,
            )

    # Save or return
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid_image.save(output_path)
        return None
    return grid_image
