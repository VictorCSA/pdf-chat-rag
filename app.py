import chainlit as cl
from dotenv import load_dotenv
from rag import build_chain

load_dotenv()


@cl.on_chat_start
async def start():
    files = await cl.AskFileMessage(
        content="Send a PDF to get started.",
        accept=["application/pdf"],
        max_size_mb=20,
    ).send()

    if files is None:
        await cl.Message(content="No file received. Please refresh and try again.").send()
        return

    pdf = files[0]
    msg = cl.Message(content="Indexing your PDF, please wait...")
    await msg.send()

    chain, retriever = build_chain(pdf.path)
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", retriever)

    msg.content = f"**{pdf.name}** is ready. Ask me anything about it."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")

    if chain is None:
        await cl.Message(content="Please upload a PDF first.").send()
        return

    source_docs = await cl.make_async(retriever.invoke)(message.content)
    answer = await cl.make_async(chain.invoke)(message.content)

    source_lines = "\n\n---\n**Sources:**\n"
    for doc in source_docs[:3]:
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:120].strip()
        source_lines += f"- Page {page + 1}: _{snippet}..._\n"

    await cl.Message(content=answer + source_lines).send()