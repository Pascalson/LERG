import streamlit as st
from annotated_text import annotated_text
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

@st.cache(allow_output_mutation=True)
def get_explain(x,y):
    local_exp = LERG(model.forward, x, y, PM.perturb_inputs, model.tokenizer)
    return local_exp.get_local_exp()

def write_header():
    st.title('Local Explanation of Dialogue Response Generation')
    st.markdown('''
        - A simple playground to understand local explanation of dialogue response generation.
        - This demo is powered by a fined-tuned GPT language model on DailyDialog.
        - For more information, please refer to [codes](https://github.com/Pascalson/LERG) and [paper](https://arxiv.org/pdf/2106.06528.pdf).
    ''')

def query_example(example_id):
    examples = [
        ("Hello!","Hi! How are you?"),
        ("Are you busy tomorrow morning?","I'm free, what's up?"),
        ("Thank you. Oh, it's very delicious.","Thank you for your praise."),
        ("What dressing would you like on the salad?","French dressing, please."),
        ("They want the government to reduce the price of gasoline.","It's really a hot potato."),
    ]
    return examples[example_id-1]

def write_ui():

    """
    add_selectbox = st.sidebar.selectbox(
        "Take a look at some examples",
        ("Email", "Home phone", "Mobile phone")
    )
    """
    example_id = st.sidebar.slider("Take a look at an example: ", 1, 5, 1, 1)
    example_input, example_response = query_example(example_id)

    #input_txt = st.text_input('Enter the first sentence in a conversation', value="Hello!")
    #response_txt = st.text_input('Enter the next sentence', value="Hi! How are you?")
    input_txt = st.text_input('Enter the first sentence in a conversation', value=example_input)
    response_txt = st.text_input('Enter the next sentence', value=example_response)
    if st.button("Explain this dialogue!"):
        if not input_txt or not response_txt:
            return
        with st.spinner('Wait for it...'):
            phi_set, phi_map, x_components, y_components = get_explain(input_txt, response_txt)
        st.success("Done!")
        st.write("- The complete saliency map")
        h = 3
        w = h/len(y_components) * len(x_components)
        col1, space = st.beta_columns([w,20-w])
        with col1:
            plot_interactions(phi_map,x_components,y_components)
        st.write("- The highlighed text")
        color_txt(phi_map,x_components,y_components)
    else:
        return

def set_model():
    from perturbation_models import RandomPM
    from RG_explainers import LERG_SHAP_log
    from target_models import GPT
    global model
    global LERG
    global PM
    PM = RandomPM(denoising=False)
    LERG = LERG_SHAP_log
    model = GPT()

def color_txt(phi_map,x,y):
    values = np.around([[phi_map[(i,j)].item() for i in range(len(x))] for j in range(len(y))], decimals=2)
    array = values.flatten()
    array.sort()
    topk_v = array[-1] * 0.5
    color_x = []
    color_y = []
    for i in range(len(x)):
        for j in range(len(y)):
            if phi_map[(i,j)].item() >= topk_v:
                if i not in color_x:
                    color_x.append(i)
                if j not in color_y:
                    color_y.append(j)
    color_x_txt, color_y_txt = [], []
    for i, xi in enumerate(x):
        if i in color_x:
            color_x_txt.append((xi,"","#faa"))
    for j, yj in enumerate(y):
        if j in color_y:
            color_y_txt.append((yj,"","#fea"))
    annotated_text(*(color_x_txt+[" in the input corresponds to "] + color_y_txt + [" in the response."]))

def plot_interactions(phi_map,x,y):
    values = np.around([[phi_map[(i,j)].item() for i in range(len(x))] for j in range(len(y))], decimals=2)
    fig = plt.figure()#figsize=(w,h))
    ax = plt.axes()
    im = ax.imshow(values, cmap=plt.get_cmap('Reds'))

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    x = [w[:-4] for w in x]
    y = [w[:-4] for w in y]
    ax.set_xticklabels(x, fontsize=11)
    ax.set_yticklabels(y, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, values[i, j],
                           ha="center", va="center", color="w")
    st.pyplot(fig) 
    plt.close()

if __name__ == '__main__':
    st.set_page_config(page_title='Local Explanation of Dialogue Response Generation', layout='wide')
    set_model()
    write_header()
    write_ui()
