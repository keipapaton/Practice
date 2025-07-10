import sys
import os
import cv2
import numpy as np
import sympy
import easyocr
from PIL import Image

# PyTorchバージョン確認
try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('PyTorchのバージョン確認でエラー:', e)

# 1. 画像ファイルのパスをユーザー入力
img_path = "Data/" + input('画像ファイルのパスを入力してください: ')
if not os.path.exists(img_path):
    print('ファイルが見つかりません:', img_path)
    sys.exit(1)
try:
    screenshot = Image.open(img_path)
    print(f'画像を読み込みました: {img_path}')
except Exception as e:
    print('画像の読み込みに失敗しました:', e)
    sys.exit(1)

# 2. Tesseract & EasyOCRで文字認識（前処理付き）
tesseract_text = ''
easyocr_text = ''
try:
    import pytesseract
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError('cv2で画像が読み込めませんでした')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 前処理画像を保存（デバッグ用）
    debug_img_path = os.path.splitext(img_path)[0] + '_preproc.png'
    cv2.imwrite(debug_img_path, thresh)
    print(f'前処理画像を保存しました: {debug_img_path}')
    print('Tesseract OCR推論開始')
    custom_config = '--psm 6'
    tesseract_text = pytesseract.image_to_string(thresh, lang='jpn+eng', config=custom_config)
    print('Tesseract OCR推論完了')
    print('Tesseract認識テキスト:')
    print(tesseract_text)
    if not tesseract_text.strip():
        print('警告: Tesseractの認識結果が空です')
    # EasyOCR
    try:
        img_for_easyocr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        print('EasyOCRリーダー初期化開始')
        reader = easyocr.Reader(['ja', 'en'])
        print('EasyOCRリーダー初期化完了')
        result_easyocr_detail = reader.readtext(img_for_easyocr, detail=1)
        print('EasyOCR詳細データ:', result_easyocr_detail)
        result_easyocr = [r[1] for r in result_easyocr_detail]
        print('EasyOCR推論完了')
        print('EasyOCRテキストリスト:', result_easyocr)
        if not result_easyocr:
            print('警告: EasyOCRの認識結果が空です')
        easyocr_text = '\n'.join(result_easyocr)
        print('EasyOCR認識テキスト:')
        print(easyocr_text)
    except Exception as e:
        import traceback
        print('EasyOCRでエラー:', e)
        traceback.print_exc()
        print('EasyOCR例外発生により終了します')
except Exception as e:
    import traceback
    print('Tesseractでエラー:', e)
    traceback.print_exc()
    print('例外発生により終了します')
    sys.exit(1)

# 以降、text変数にはTesseractの結果を使う（必要に応じて切替可）
text = tesseract_text

# 3. 数式・図表・知識系の意味解析
has_formula = any(c in text for c in ['=', '+', '-', '*', '/', '^'])
has_table = any(word in text for word in ['表', 'table', 'Table'])
has_figure = any(word in text for word in ['図', 'figure', 'Figure'])

if has_formula:
    try:
        expr = sympy.sympify(text)
        result = sympy.simplify(expr)
        print('数式の簡易評価:', result)
    except Exception as e:
        print('数式解析に失敗:', e)
else:
    print('数式は検出されませんでした。')

if has_table:
    print('表（テーブル）が含まれている可能性があります。')
if has_figure:
    print('図（グラフ・図表）が含まれている可能性があります。')

# 4. 知識系・計算系の検索/生成（例: Wikipedia検索）
import webbrowser
import urllib.parse
query = text.strip().replace('\n', ' ')
if query:
    print('関連するWikipediaページをブラウザで検索します...')
    url = f'https://ja.wikipedia.org/wiki/Special:Search?search={urllib.parse.quote(query)}'
    webbrowser.open(url)
else:
    print('検索クエリがありません。')

# 5. コンソールに回答を出力
print('\n--- 解析結果まとめ ---')
print('認識テキスト:')
print(text)
if has_formula:
    print('数式が含まれています。')
if has_table:
    print('表が含まれています。')
if has_figure:
    print('図が含まれています。')
