# Mini Energy Class Room Digital Twin

目的：授業中24〜28℃の室温条件を満たしながら、冷房を低炭素な時間帯へ移す運転の効果と限界を比較。

## 起動

`pip install -r requirements.txt` の後、`streamlit run app.py`。
同じMacでは「起動.command」も使用可能。

## 授業での比較

1. 高断熱・授業前が低炭素の初期条件で、電力量とCO₂の差を比較。
2. 排出係数を一定・逆転に変更。電力量の増加を伴う予冷が常に削減につながるわけではないことを確認。
3. 断熱性能を変更。外皮からの熱流入と予冷効果の関係を確認。
4. 授業中の出力上限を下げ、室温条件の未達を確認。未達案を削減策として評価しない。
5. 「条件による効果の違い」で9条件を比較。建物と電力供給の状況に応じた判断を検討。

通常運転も授業開始から同じ温度範囲を満たすよう開始時刻を探索。予冷前から授業終了までの全電力を積算。CO₂は時間ごとの電力量×排出係数を合計。

外皮性能と熱容量は別々の設定。換気による熱流入は断熱性能を変えても不変。

実測に接続する前の簡易熱収支モデル。排出係数は条件比較用で実在の地域・日の予測ではない。断熱改修に伴う建設・製造の排出量、限界排出係数による系統全体の因果的削減は未評価。授業終了時に残る冷熱は未精算（両案の終端温度をモデル説明に表示）。湿度・空気と壁の温度差・機器起動損失等は省略。

参考：Berkeley Lab, Demand Shifting With Thermal Mass in Large Commercial Buildings
https://eta-publications.lbl.gov/publications/demand-shifting-thermal-mass-large-0

検証：`python -m unittest test_model.py`（熱・電力量・CO₂収支、基準案の室温充足と最遅開始、排出係数による効果の逆転、条件未達判定）。

## Streamlit Cloudへの反映

app.pyに計算モデルを統合。Cloud側のenergy/05_class_room/app.pyをこの版で置換。別ファイルmodel.pyの更新は不要。requirements.txtも同じフォルダに配置。
