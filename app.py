import streamlit as st

import pandas as pd
import numpy as np


def load_item1_sample_result():
    with open('data/item1_sample_result.txt', 'r', encoding='utf-8') as rf:
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


def load_item3_sample_result():
    with open('data/item3_sample_result.txt', 'r', encoding='utf-8') as rf:
        lines = rf.readlines()

        tword_dict = {}

        for lidx, line in enumerate(lines):
            result, true, pred, prev_sent, next_sent = (line.rstrip()).split('\t')
            tword_dict[f'example_{lidx}'] = {
                'result': result,
                'true_label': true,
                'pred_label': pred,
                'prev_sent': prev_sent,
                'next_sent': next_sent
            }

    return tword_dict


def load_vocab():
    label_dict = {}
    with open('data/item1_label.vocab', 'r', encoding='utf-8') as rf:
        for line in rf.readlines():
            index, label = (line.rstrip()).split('\t')
            label_dict[index] = label

    return label_dict


def main():
    st.title('Paper Assistant 2022')
    menu = ['Abstract Analysis', 'Transition Word Recommend']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Abstract Analysis':
        abs_dict = load_item1_sample_result()
        label_dict = load_vocab()

        st.subheader('Abstract Analysis')
        with st.expander('SPA 2022: Abstract Class Label'):
            st.markdown("""
                * Introduction:  현재 문헌에 대한 reference, 주제의 중요성, 지식 격차 식별
                * Aims: 현재 연구의 목표 
                * Method: 사용된 방법에 대한 설명
                * Results: 주요 연구 결과에 대한 설명
                * Discussion: 연구 결과의 의미, 현재 연구의 가치에 대한 내용
                * Extra: 연구 자료 등의 전달을 위한 url과 같은 추가 정보
            """)

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

    elif choice == 'Transition Word Recommend':
        tword_dict = load_item3_sample_result()

        tword_label_dict = {
            'Additive': 'moreover, in addition, furthermore, as well as, in fact, as a matter of fact, \
                        additionally, that is, in other words, put another way, put it another way, what this means is,\
                        this means, specifically, namely, for example, for instance, to illustrate, in particular, \
                        one example, particularly, notably, especially',
            'Adversative': 'yet, nevertheless, despite, in spite of, although, even though, conversely, on the contrary, \
                            when in fact, by way of contrast, nonetheless, regardless, admittedly, even so, \
                            be that as it my, however, in contrast, unlike, in contrast to,  while, whereas',
            'Causal': 'thus, hence, therefore, as a result, consequently, for this reason, as a consequence, so much \
                        that, accordingly, due to, owing to, because of, on account of, since, for the reason that',
            'Sequence': 'firstly, first of all, secondly, thirdly, finally, in conclusion, initially, to start with, \
                    in the first place, in the second place, in the third place',
        }

        st.subheader('Transition word recommend')
        with st.expander('SPA 2022: Transition Word Class Label'):
            st.markdown("""
                        * Additive:  signal that you are adding or referencing information
                        * Adversative: indicate conflict or disagreement
                        * Causal: point to consequences and show cause-and-effect relationships
                        * Sequence: clarify the sequence of information and overall structure of the paper
                    """)

        with st.form(key='transition_word'):

            choice = st.selectbox(
                'select transition word sentence example from the list',
                (tword_dict.keys())
            )
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            a_dict = tword_dict[choice]
            prev_sent = a_dict['prev_sent']
            next_sent = a_dict['next_sent']

            st.subheader('prev sentence')
            st.write(prev_sent)

            st.subheader('next sentence')
            st.write(next_sent)

            st.write('-------------------------------')
            st.subheader('Result')
            pred_label = a_dict['pred_label']
            st.markdown(f'{prev_sent}<span style="background-color:#0E4D92;color:white; border-radius:2%;margin: 0px 5px; padding: 2px 5px; border-radius: 4px">{pred_label}</span>{next_sent}', unsafe_allow_html=True)


            st.write('-------------------------------')
            st.subheader('Recommend transition words')
            st.subheader(pred_label)
            st.info(tword_label_dict[pred_label])



if __name__ == '__main__':
    main()