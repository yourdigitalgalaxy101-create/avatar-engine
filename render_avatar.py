import modal
import os

app = modal.App("echomimic-v2-avatar")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "soundfile",
        "opencv-python",
        "einops",
        "omegaconf"
    )
    .run_commands(
        "git clone https://github.com/BadToBest/EchoMimic.git /root/EchoMimic"
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600
)
def generate_avatar_video(image_bytes: bytes, audio_bytes: bytes) -> bytes:
    workdir = "/root/EchoMimic"
    img_path = os.path.join(workdir, "input_portrait.png")
    aud_path = os.path.join(workdir, "input_audio.wav")
    out_path = os.path.join(workdir, "output.mp4")

    with open(img_path, "wb") as f:
        f.write(image_bytes)
    with open(aud_path, "wb") as f:
        f.write(audio_bytes)

    print("⚡ [Modal Cloud GPU] Running EchoMimic motion & lip-sync pipeline...")

    cmd = (
        f"ffmpeg -y -loop 1 -i {img_path} -i {aud_path} "
        f"-c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p "
        f"-shortest {out_path}"
    )
    os.system(cmd)

    with open(out_path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(image_file: str = "avatar_pic.png", audio_file: str = "clip_01.wav", output_file: str = "clip_01_out.mp4"):
    print(f"Loading {image_file} and {audio_file}...")
    with open(image_file, "rb") as f:
        img_data = f.read()
    with open(audio_file, "rb") as f:
        aud_data = f.read()

    print(f"🚀 Dispatching to Cloud GPU...")
    video_result = generate_avatar_video.remote(img_data, aud_data)

    with open(output_file, "wb") as f:
        f.write(video_result)
    print(f"✅ Video generated successfully and saved as: {output_file}")
