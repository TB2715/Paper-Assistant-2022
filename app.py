import streamlit as st

import pandas as pd
import numpy as np

import joblib


def load_sample_result():
    with open('./data/sample_result.txt','r',encoding='utf-8') as rf:
        lines = rf.readlines()

        a_sent_idx = 0
        a_title = ''
        abs_dict = {}

        for line in lines:
            if line[0] == '-':
                a_sent_idx = 0
                a_title = ''
                continue

            if a_sent_idx == 0:
                a_title = line
                abs_dict[a_title] = []
            else:
                abs_dict[a_title].append(line)
            a_sent_idx += 1

    return abs_dict


def load_vocab():
    label_dict = {}
    with open('./data/label.vocab', 'r', encoding='utf-8') as rf:
        for line in rf.readlines():
            index, label = (line.rstrip()).split('\t')
            label_dict[index] = label

    return label_dict

def main():
    st.title('Paper Assistant 2022')
    menu = ['Abstract', 'Transition Word']
    choice = st.sidebar.selectbox('Menu', menu)

    abs_dict = load_sample_result()
    label_dict = load_vocab()

    if choice == 'Abstract':
        st.subheader('Abstract Analysis')
        with st.expander('SPA 2022: Abstract Class Label'):
            st.markdown('* ')

        with st.form(key='abstract_sentence'):
            choice = st.selectbox(
                'select a abstract from the list',
                (abs_dict.keys())
            )
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col = st.columns(len(abs_dict[choice]))

            temp_dict = {}
            for a_s_idx, a_sent in enumerate(abs_dict[choice]):
                sentence, true, pred = (a_sent.rstrip()).split('\t')
                true_label = label_dict[true]
                pred_label = label_dict[pred]
                result = 'True' if true == pred else 'False'
                temp_dict[a_s_idx] = [sentence, true_label, pred_label, result]

            temp_df = pd.DataFrame(temp_dict, index=['sentence', 'true_label', 'predict_label', 'result'])
            temp_df = temp_df.transpose()
            st.table(temp_df)
                # with col[a_s_idx]:
                #     st.success('Original Text')
                #
                #     st.write(abs_dict[choice][a_s_idx])
                #
                #     st.success('Answer')
                #     st.write()

    elif choice == 'Transition Word':
        st.subheader('Transition word recommend')


if __name__ == '__main__':
    main()