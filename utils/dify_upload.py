import mimetypes
import os
import aiohttp


async def upload_image_file(file_path: str):
    # 确保 base_url 没有末尾斜杠，防止拼接出双斜杠 (虽然 http 不太敏感，但为了严谨)
    base_url = os.getenv('base_url').rstrip('/')
    upload_url = f"http://{base_url}:21000/v1/files/upload"

    headers = {
        "Authorization": f"Bearer {os.getenv('api_key')}"
    }

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)
    file_name = os.path.basename(file_path)

    # 增加超时时间，防止大文件上传中断
    timeout = aiohttp.ClientTimeout(total=120)

    # 修正点：将文件读取为 bytes，避免 chunked 传输
    with open(file_path, "rb") as f:
        file_content = f.read()

    data = aiohttp.FormData()
    data.add_field(
        name="file",
        value=file_content,  # 传入 bytes 数据
        filename=file_name,
        content_type=mime_type or "application/octet-stream"
    )
    data.add_field("user", "abc-123")

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(upload_url, headers=headers, data=data) as response:
                text = await response.json()

                if response.status in (200, 201):
                    return text.get("id")
                else:
                    print(f"Upload Failed. Status: {response.status}, Response: {text}")
                    raise RuntimeError(f"Upload failed: {response.status} - {text}")

    except aiohttp.ClientOSError as e:
        # 捕获连接重置错误，提供更有用的调试信息
        print(f"Network error interacting with {upload_url}: {e}")
        raise


async def run_workflow(file_id, src_age_estimate, src_recognition_results, response_mode="blocking"):
    workflow_url = f"http://{os.getenv('base_url')}:{os.getenv('dify_port')}/v1/workflows/run"
    headers = {
        "Authorization": f"Bearer {os.getenv('api_key')}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": {
            "image_upload": {
                        "transfer_method": "local_file",
                        "upload_file_id": file_id,
                        "type": "image"},
            "src_age_estimate": src_age_estimate,
            "src_recognition_results": src_recognition_results
        },
        "response_mode": response_mode,
        "user":  os.getenv("app_user")
    }

    try:
        print("运行工作流...")
        async with aiohttp.ClientSession() as session:
            async with session.post(workflow_url, headers=headers, json=data) as response:
                resp_json = await response.json()
                if response.status == 200:
                    print("工作流执行成功")
                    return resp_json["data"]["outputs"]
                else:
                    print(f"工作流执行失败，状态码: {response.status}")
                    print(resp_json)
                    return False
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

# 示例调用：
# asyncio.run(run_workflow("your_file_id", "user123", "识别图片内容", {"sensor": "value"}))
