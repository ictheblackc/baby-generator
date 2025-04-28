import aiohttp
from datetime import datetime

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


async def generate_child_image(client, prompt: str, user_dir) -> str:
    """Generate image using OpenAI DALL-E model."""
    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = response.data[0].url

    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch image: {resp.status}")
            image_data = await resp.read()
    
    user_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = user_dir / f"child_{timestamp}.png"
    with open(image_path, "wb") as f:
        f.write(image_data)

    return str(image_path)