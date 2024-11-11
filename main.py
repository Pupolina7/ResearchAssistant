import gradio as gr
from functionality import get_collection, predict

def read_file(file):
    if file is not None:
        with open(file.name, 'r') as f:
            file_data = f.read()
        return file_data
    else:
        return None

with gr.Blocks() as demo:
    gr.Markdown("# **📝 AI-powered Academic Research Assistant 📝**")
    gr.Markdown("""**AI-powered Academic Research Assistant** is a tool which helps to 
                ensure the *correct grammar* and *academic style* in the scientific papers.

                It also could help with *writing needed parts* or *proposing possible ideas*
                for describing what you want in appropriate way.

                ## 📥 Down bellow you should choose appropriate parameters for your goals and then wait a little for the responce!""")

    gr.Markdown('📨 Write the text you what to expand or upload corresponding text file.')    

    with gr.Tab('Write Text 📖'):
        gr.Markdown("⚙️ *Hint*: to ensure more effective work of 'Fix Academic Style' try to make your sentences not too long (<= 20 words).")
        input_prompt = gr.Textbox(label='Initial Text 📝',
                            placeholder='Write here your research text!',
                            lines=9,)
    with gr.Tab('Upload File 📩'):
        gr.Markdown("⚙️ *Hint*: to ensure more effective work of 'Fix Academic Style' try to make your sentences not too long (<= 20 words).")
        txt_file = gr.File(file_types=['text',], label='Upload Text File',)
        txt_file.change(read_file, inputs=txt_file, outputs=input_prompt)

            
    gr.Markdown('✏️ Fill parameters for your needs')
    with gr.Row(variant='panel', equal_height=True):
        request_goal = gr.Radio(label='🤔 Specify the purpose of your request.',
                                info="Pick one:",
                                choices=['Write Text (Part)', 'Fix Academic Style', 'Fix Grammar', ],
                                value='Write Text (Part)',)
        
        with gr.Accordion("❗️ In case you need to Write Text (Part) choose appropriate option!", open=False):
            part_to_write = gr.CheckboxGroup(label="""📋 What part for Assistant to write? (here you need to 
                                             specify what part of your research you need to complete.)""", 
                                            info="""You may chose as many as needed:""",
                                            choices=['Abstract', 'Introduction', 
                                                     'Methodology', 'Discussion', 'Conclusion', 'Full Text',],
                                            value='Abstract',)
    
    with gr.Row(equal_height=True):
        submit_btn = gr.Button('Confirm! ✅')
        clear_btn = gr.Button('Clear ❌', min_width=611)

    gr.Markdown('##### 📌 Assistant Responce')
    gr.Markdown("In case you did not satisfy with the responce try to paraphrase!")

    responce = gr.Textbox(label="Generated Text 👨🏼‍💻", 
                        info="""You may face some page jumps, it is a bug which will be fixed. Just wait for the completion of text generation. 
                        Sorry for inconvenience(""",
                        lines=9, 
                        placeholder='Here the generated text will appear!', 
                        show_label=True,
                        show_copy_button=True,
                        autofocus=True, 
                        autoscroll=True,)
    
    submit_btn.click(fn=predict, 
                    inputs=[request_goal, part_to_write, input_prompt,], 
                    outputs=[responce], 
                    scroll_to_output=True, 
                    queue=True)
    clear_btn.click(lambda: (None, None, 'Write Text (Part)', 'Abstract', None), None, 
                outputs=[input_prompt, txt_file, request_goal, part_to_write, responce])

if __name__ == "__main__":
    get_collection()
    demo.launch(share=True)
