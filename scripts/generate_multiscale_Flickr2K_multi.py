import argparse
import glob
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_image(path, output_dir):
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400

    basename = os.path.splitext(os.path.basename(path))[0]
    try:
        img = Image.open(path).convert("L")
        width, height = img.size

        # 多尺度缩放
        for idx, scale in enumerate(scale_list):
            rlt = img.resize(
                (int(width * scale), int(height * scale)), resample=Image.LANCZOS
            )
            rlt.save(os.path.join(output_dir, f"{basename}T{idx}.png"))

        # 生成最小边为400的图像
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)

        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(output_dir, f"{basename}T{len(scale_list)}.png"))

        print(f"[OK] {basename}")
    except Exception as e:
        print(f"[ERROR] {basename}: {e}")


def main(args):
    path_list = sorted(glob.glob(os.path.join(args.input, "*")))
    os.makedirs(args.output, exist_ok=True)

    # 控制线程数（可根据CPU核心数调整）
    num_threads = min(24, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_image, path, args.output) for path in path_list]
        for f in as_completed(futures):
            pass  # 可在此添加进度统计等


if __name__ == "__main__":
    """Generate multi-scale versions for GT images with LANCZOS resampling (multi-threaded)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="datasets/Flickr2K/Flickr2K_HR", help="Input folder")
    parser.add_argument("--output", type=str, default="datasets/Flickr2K/Flickr2K_multiscale_gray", help="Output folder")
    args = parser.parse_args()

    main(args)
