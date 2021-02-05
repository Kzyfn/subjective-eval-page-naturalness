## coding: UTF-8
# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from os import remove
from os.path import join
import pandas as pd
import random
import string
import pickle

from operator import itemgetter

def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)


def int2str(num):
    return '0'*(4-len(str(num))) + str(num)

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
models = ['decision_tree_nc2', 'decision_tree_wo_accent', 'with_accent_z', 'wo_accent_z']
bcps = ['OSAKA864_{}.wav'.format(int2str(i*13+2)) for i in range(67)]


# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理

model_a = '/static/wav/decision_tree_nc2/'
model_b = '/static/wav/decision_tree_wo_accent/'

@app.route('/', methods=['GET'])
def index():
    """
    ここでランダムにbcpを20個選んできて20個とも同じ画面に表示
    """
    n = 5
    chosen_indices = np.random.randint(0, 67, n)
    chosen_bcps = []
    for idx in chosen_indices:
        chosen_bcps.append(bcps[idx])
    
    model_a_order = []
    model_b_order = []

    model_a_style = []
    model_b_style = []
    for i in range(n):
        if np.random.rand() > 0.5:
            style_a = "flex-direction:column;align-items:center;order:0;"
            style_b = "flex-direction:column;align-items:center;order:1;"

            model_a_order.append(1)
            model_b_order.append(2)
            model_a_style.append(style_a)
            model_b_style.append(style_b)
        else:
            style_a = "flex-direction:column;align-items:center;order:1;"
            style_b = "flex-direction:column;align-items:center;order:0;"

            model_b_order.append(2)
            model_a_order.append(1)
            model_a_style.append(style_a)
            model_b_style.append(style_b)


    random.shuffle(chosen_bcps)
    # index.html をレンダリングする


    return render_template('index.html', model_a=model_a, model_b=model_b, wav_files_and_orders=zip(chosen_bcps, model_a_order, model_b_order, model_a_style, model_b_style) )


@app.route('/done', methods=['GET', 'POST'])
def done():
    if request.method == 'POST':
        result_path = randomname(20)#結果ファイルの識別子
        result_dict = {'file':[], 'preferred_model':[]}

        """
        request.form: {'/static/wav/{model}/{bcp}.wav': {値},... }みたいなものが入ってる 
        ex. {'/static/wav/model1/OSAKA1200_0001.wav': 0, '/static/wav/model2/OSAKA1200_0001.wav': 4, ...}
        """
        models = [model_a, model_b]


        for key in request.form.keys():
            result_dict['preferred_model'].append( models[int(request.form[key])] )
            result_dict['file'].append( key )


        pd.DataFrame(result_dict).to_csv('static/results/{}.csv'.format(result_path), index=None)
        

        return render_template('done.html', result_path = result_path)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0', port=5000) # どこからでもアクセス可能に
