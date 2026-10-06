# Mini PV Panel Digital Twin

起動：`streamlit run app.py`

左側で季節・傾斜角・方位を設定、右側で太陽とパネルの関係、１日の発電出力、発電量を確認。時刻は図の確認位置のみを変更。

## 教員による短いデモ

1. 南向きで夏至と冬至を切り替え、傾斜角を0°・30°・60°に変更。季節による違いを確認。
2. 同じ季節で東向きと西向きを比較。最大出力の時間帯の違いを確認。
3. 「角度を考慮しない計算を重ねる」を選択。水平面の日射をそのまま使うと設置角度への応答を捉えられないことを確認。

## 計算条件

東京付近（緯度35.7°）、25 m²、効率20%固定。５分間隔。時刻は太陽時（南中12時）で、日本標準時ではない。季節の代表日から赤緯を簡易計算し、太陽方向とパネル法線の内積から入射角を計算。

教材用日射：太陽高度αに対して DNI＝900 exp(−0.16/sinα)、DHI＝100 sqrt(sinα)（日中のみ）。晴天傾向を示すための仮定で、実測・気象予測や厳密な晴天モデルではない。

パネル面日射＝直達成分＋等方的な天空散乱成分＋地表反射成分（反射率0.2）。発電出力は直流出力。影・温度による効率変化・機器損失は含まない。比較線は同じ水平面日射からの計算で傾斜0°と一致。

計算関係の参考：
- https://github.com/pvlib/pvlib-python/blob/main/pvlib/solarposition.py
- https://github.com/pvlib/pvlib-python/blob/main/pvlib/irradiance.py
- https://pvlib-python.readthedocs.io/en/v0.11.1/reference/generated/pvlib.irradiance.isotropic.html

追加データ・SciPy・APIキー不要。
