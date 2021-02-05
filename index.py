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



@app.route('/', methods=['GET'])
def index():
    """
    ここでランダムにbcpを20個選んできて20個とも同じ画面に表示
    """

    chosen_bcps = np.random.choice(bcps, 5, replace=False)

    files = []
    for model in models:
        for chosen_bcp in chosen_bcps:
            files.append(join('/static/wav', model, chosen_bcp))

    random.shuffle(files)
    # index.html をレンダリングする

    return render_template('index.html', wav_files=files)


@app.route('/done', methods=['GET', 'POST'])
def done():
    if request.method == 'POST':
        result_path = randomname(20)#結果ファイルの識別子
        result_dict = {model:{} for model in models}#結果格納用辞書

        """
        request.form: {'/static/wav/{model}/{bcp}.wav': {値},... }みたいなものが入ってる 
        ex. {'/static/wav/model1/OSAKA1200_0001.wav': 0, '/static/wav/model2/OSAKA1200_0001.wav': 4, ...}
        """

        for key in request.form.keys():
            for model in models:
                if model in key:
                   result_dict[model][key[key.rfind('/')+1:]] =  request.form[key]

        pd.DataFrame(result_dict).to_csv('static/results/{}.csv'.format(result_path))

        return render_template('done.html', result_path = result_path)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0', port=80) # どこからでもアクセス可能に
