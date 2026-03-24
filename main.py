import os
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class SummaryOutput(BaseModel):
    insight: str = Field(description="一句话总结")
    info_list: list[str] = Field(description="列举这个机制的优缺点，3-5条即可")
    usage: list[str] = Field(description="列举出该机制的应用场景，2-5个即可")


def demo_prompt_template() -> None:
    print("=== PromptTemplate 输入 ===")
    role = input("请输入角色 role: ").strip()
    topic = input("请输入主题 topic: ").strip()
  

    template = PromptTemplate.from_template(
        "你是一位{role}，请用简洁清晰的方式解释：{topic}。"
    )
    prompt_text = template.format(role=role, topic=topic)

    print("=== PromptTemplate 输出 ===")
    print(prompt_text)


def demo_chat_prompt_template() -> ChatPromptValue:
    print("\n=== ChatPromptTemplate 输入 ===")
    role = input("请输入角色 role: ").strip()
    style = input("请输入回答风格 style: ").strip()
    topic = input("请输入主题 topic: ").strip()

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位{role}，回答风格要求：{style}。"),
            ("human", "请解释一下：{topic}"),
        ]
    )

    prompt_value = chat_template.invoke(
        {
            "role": role,
            "style": style,
            "topic": topic,
        }
    )

    print("\n=== ChatPromptTemplate 输出 ===")
    for msg in prompt_value.messages:
        print(f"[{msg.type}] {msg.content}")

    return prompt_value


def call_openai_with_chat_prompt(chat_prompt: ChatPromptValue) -> None:
    print("\n=== OpenAI 调用 ===")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("未检测到 OPENAI_API_KEY，已跳过模型调用。")
        print("请在项目根目录创建 .env 并写入：OPENAI_API_KEY=你的密钥")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=SummaryOutput)
    format_instructions = parser.get_format_instructions()

    messages = list(chat_prompt.messages)
    messages.append(
        HumanMessage(
            content=(
                "请基于上面的对话内容，输出结构化总结。\n"
                "必须严格遵循以下格式说明，不要输出额外文本：\n"
                f"{format_instructions}"
            )
        )
    )
    response = llm.invoke(messages)

    print("\n=== OpenAI 原始回复 ===")
    print(response.content)

    try:
        parsed_output = parser.parse(response.content)
        print("\n=== PydanticOutputParser 解析结果 ===")
        print(parsed_output)
        print("\n=== JSON 输出 ===")
        print(json.dumps(parsed_output.model_dump(), ensure_ascii=False, indent=2))
    except Exception as error:
        print("\n结构化解析失败：", error)
        print("请检查模型输出是否为符合格式说明的 JSON。")


def call_openai_with_chat_prompt_lcel(chat_prompt: ChatPromptValue) -> None:
    print("\n=== OpenAI 调用（LCEL）===")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("未检测到 OPENAI_API_KEY，已跳过模型调用。")
        print("请在项目根目录创建 .env 并写入：OPENAI_API_KEY=你的密钥")
        return

    lcel_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_messages"),
            (
                "human",
                "请基于上面的对话内容，输出结构化总结。\n"
                "必须严格遵循以下格式说明，不要输出额外文本：\n"
                "{format_instructions}",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    parser = PydanticOutputParser(pydantic_object=SummaryOutput)

    chain = lcel_prompt | llm | parser

    try:
        parsed_output = chain.invoke(
            {
                "chat_messages": chat_prompt.messages,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        print("\n=== LCEL 解析结果 ===")
        print(parsed_output)
        print("\n=== JSON 输出 ===")
        print(json.dumps(parsed_output.model_dump(), ensure_ascii=False, indent=2))
    except Exception as error:
        print("\nLCEL 结构化解析失败：", error)
        print("请检查 API Key、模型可用性，或稍后重试。")


def main() -> None:
    # Auto-load variables from .env in project root.
    load_dotenv()

    # demo_prompt_template()
    chat_prompt = demo_chat_prompt_template()
    should_call_llm = input("\n是否调用 OpenAI 模型？(y/n): ").strip().lower()
    if should_call_llm in {"y", "yes"}:
        method = input("请选择调用方式：1=老方法 2=LCEL（默认1）: ").strip()
        if method == "2":
            call_openai_with_chat_prompt_lcel(chat_prompt)
        else:
            call_openai_with_chat_prompt(chat_prompt)
    else:
        print("已跳过 OpenAI 模型调用。")


if __name__ == "__main__":
    main()