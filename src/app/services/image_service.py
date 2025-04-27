async def generate_prompt(desc: dict, gender: str) -> str:
    """Build the prompt for the AI image generation based on attributes."""
    return (
        f"A photorealistic studio portrait of a {gender} child. "
        f"The child has skin tone similar to the {desc['race']}, "
        f"eyes approximately the color {desc['eye_color_rgb']}, "
        f"hair color around {desc['hair_color_rgb']}, "
        f"and a facial expression resembling {desc['emotion']}. "
        f"Studio lighting, high resolution, realistic style."
    )


async def generate_child_image(client, prompt: str) -> str:
    """Generate image using OpenAI DALL-E model."""
    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url
