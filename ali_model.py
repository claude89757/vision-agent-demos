from openai import OpenAI
import os
import base64



def get_tennis_action_comment(action_image_path: str, model_name: str = "qwen-vl-max-latest", action_type: str = "击球准备动作") -> str:
    """
    通过阿里云的AI模型，获取网球动作的评论
    """
    system_prompt = f"专业的网球教练，擅长对网球动作进行分析和评价。请结合照片网球运动员的{action_type}的细节，给出评价。格式如下：\n" \
                    f"评分等级：S|A|B|C\n" \
                    f"动作评价：10字以内\n" \
                    f"动作建议：10字以内(如果比较完美，可以不给出建议)"

    #  base 64 编码格式
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # 将xxxx/test.png替换为你本地图像的绝对路径
    base64_image = encode_image(action_image_path)
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 获取动作评价  
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    action_image_path = "/Users/claude89757/github/vision-agent-demos/output/Roger_Federer_20250324_214047/prep_frame.jpg"
    comment = get_tennis_action_comment(action_image_path, model_name="qwen-vl-max-2025-01-25")
    print(comment)