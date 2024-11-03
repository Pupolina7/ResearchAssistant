import gradio as gr
from functionality import handle_user_prompt, get_collection

def read_file(file):
    # Read the file content
    if file is not None:
        with open(file.name, 'r') as f:
            file_data = f.read()
        return file_data
    else:
        return None

with gr.Blocks() as demo:
    gr.Markdown("# **📝AI-powered Academic Research Assistant📝**")
    gr.Markdown("### 🗒️ Description for AI-powered Academic Research Assistant")

    gr.Markdown('Write the text you what to expand or upload corresponding text file.')

    with gr.Tab('Write Text'):
        input_prompt = gr.Textbox(label='Initial Text',
                            placeholder='Write here your research text!',
                            lines=9,)
    with gr.Tab('Upload File'):
        # file_content = gr.Textbox(visible=False)
        txt_file = gr.File(file_types=['text','pdf',], label='Upload Text File',)
        txt_file.change(read_file, inputs=txt_file, outputs=input_prompt)

            
    gr.Markdown('✔️Fill parameters for your needs')
    with gr.Row(equal_height=True):
        request_goal = gr.Radio(label='Specify the purpose of your request.',
                                info="Pick one:",
                                choices=['Check Academic Style', 'Check Grammar', 'Write Text (Part)',],
                                value='Check Academic Style',)
        
        with gr.Accordion("In case you need to Write Text (Part) choose appropriate option!", open=False):
            part_to_write = gr.CheckboxGroup(label='What part for Assistant to write?', 
                                            info="""Here you need to specify what part of your research 
                                            you need to complete.\n You may chose as many as needed:""",
                                            choices=['Abstract', 'Introduction', 
                                                     'Methodology', 'Discussion', 'Conclusion', 'Full Text',],
                                            value='Abstract',)
    
    with gr.Row(equal_height=True):
        submit_btn = gr.Button('Confirm!')
        clear_btn = gr.Button('Clear')

    gr.Markdown('##### 📌Assistant Responce')
    responce = gr.Textbox(label="Generated Text", 
                          lines=9, 
                          placeholder='Here the generated text will appear!', 
                          show_copy_button=True)
    
    
    submit_btn.click(fn=handle_user_prompt, inputs=[request_goal, part_to_write, input_prompt,], outputs=[responce])
    clear_btn.click(lambda: (None, None, None, None, None), None, 
                    outputs=[input_prompt, txt_file, request_goal, part_to_write, responce])

if __name__ == "__main__":
    get_collection()
    demo.launch()
