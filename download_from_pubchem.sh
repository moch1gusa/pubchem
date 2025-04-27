#!/bin/bash

# --- 設定 ---

# 日付指定データ
# FTP_BASE_URL="ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Daily" 
# DATE_DIR=${1:-"2025-04-21"} 

#全データ
FTP_BASE_URL="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/" 
DATE_DIR="CURRENT-Full"

FORMAT_DIR="SDF"
# 保存フォルダ
LOCAL_OUTPUT_BASE_DIR="./pubchem_downloads"

# --- パスとURLを構築 ---
FTP_DIR_URL="${FTP_BASE_URL}/${DATE_DIR}/${FORMAT_DIR}"

LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_BASE_DIR}/${DATE_DIR}_${FORMAT_DIR}"

echo "=== PubChem Daily Downloader & Verifier ==="
echo "Source FTP Directory: ${FTP_DIR_URL}"
echo "Local Target Directory: ${LOCAL_OUTPUT_DIR}"
echo ""

# --- ローカルディレクトリ作成 ---
echo "[1/4] ローカルディレクトリを作成中..."
mkdir -p "${LOCAL_OUTPUT_DIR}"
if [ $? -ne 0 ]; then
  echo "エラー: ローカルディレクトリ (${LOCAL_OUTPUT_DIR}) の作成に失敗しました。"
  exit 1
fi
echo "ローカルディレクトリ作成完了。"
echo ""

# --- SDFファイルリストの取得 ---
echo "[2/4] FTPサーバーからSDFファイルリストを取得中..."
# wgetでディレクトリリストを取得し、grepとcutで Compound_*.sdf.gz ファイル名を抽出
# grep -P はPerl互換正規表現を使用。環境によっては利用できない場合あり。
sdf_files=$(wget -q -O - "${FTP_DIR_URL}/" | grep -oP 'href="Compound_[^"]+\.sdf\.gz"' | cut -d'"' -f2)

if [ -z "$sdf_files" ]; then
  echo "警告: ${FTP_DIR_URL}/ に 'Compound_*.sdf.gz' ファイルが見つかりませんでした。"
  echo "この日付/パスには処理するファイルがありません。"
  exit 0 # 処理対象なしで正常終了
fi
# 抽出したファイルリストを表示（確認用）
# echo "処理対象のSDFファイル:"
# echo "$sdf_files"
echo "ファイルリスト取得完了。ダウンロードを開始します。"
echo ""

# --- ファイルダウンロード ---
echo "[3/4] SDFファイルとMD5ファイルをダウンロード中..."
download_errors=0
# 抽出したファイル名を1行ずつ読み込んでループ
while IFS= read -r data_filename; do
  # 空行はスキップ
  if [ -z "$data_filename" ]; then continue; fi

  checksum_filename="${data_filename}.md5"
  data_url="${FTP_DIR_URL}/${data_filename}"
  checksum_url="${FTP_DIR_URL}/${checksum_filename}"
  local_data_path="${LOCAL_OUTPUT_DIR}/${data_filename}"
  local_checksum_path="${LOCAL_OUTPUT_DIR}/${checksum_filename}"

  # 既にファイルが存在する場合はスキップ
  if [ ! -f "$local_data_path" ]; then
      echo "  Downloading ${data_filename}..."
      wget -q -P "${LOCAL_OUTPUT_DIR}" "${data_url}"
      if [ $? -ne 0 ]; then
          echo "    エラー: ${data_filename} のダウンロードに失敗しました。"
          ((download_errors++))
          continue
      fi
  else
      echo "  Skipping (Already Exists): ${data_filename}"
  fi

  # チェックサムファイルのダウンロード
  if [ ! -f "$local_checksum_path" ]; then
      echo "  Downloading ${checksum_filename}..."
      wget -q -P "${LOCAL_OUTPUT_DIR}" "${checksum_url}"
      if [ $? -ne 0 ]; then
          echo "    エラー: ${checksum_filename} のダウンロードに失敗しました。"
          ((download_errors++))
      fi
  else
       echo "  Skipping (Already Exists): ${checksum_filename}"
  fi

done <<< "$sdf_files" # ヒアドキュメントでファイルリストをループに渡す

if [ $download_errors -gt 0 ]; then
    echo "警告: ${download_errors} 件のダウンロードエラーが発生しました。"
fi
echo "ダウンロード処理完了。"
echo ""

# --- 検証処理 ---
echo "[4/4] ダウンロードされたファイルの整合性を検証中..."
total_files=0
success_count=0
fail_count=0
missing_md5_count=0

# ローカルディレクトリ内の .sdf.gz ファイルを検索
find "${LOCAL_OUTPUT_DIR}" -maxdepth 1 -name 'Compound_*.sdf.gz' -print0 | while IFS= read -r -d $'\0' local_data_file; do
    base_filename=$(basename "$local_data_file")
    local_checksum_file="${local_data_file}.md5"
    ((total_files++))

    echo "  Verifying ${base_filename}..."

    # 対応するMD5ファイルが存在するか確認
    if [ ! -f "$local_checksum_file" ]; then
        echo "    エラー: チェックサムファイルが見つかりません (${local_checksum_file})"
        ((missing_md5_count++))
        ((fail_count++))
        continue # このファイルの検証はスキップ
    fi

    # 期待されるMD5値を取得
    expected_md5=$(cat "$local_checksum_file" | awk '{print $1}')
    if [ -z "$expected_md5" ]; then
        echo "    エラー: MD5ファイルから期待値を取得できませんでした (${local_checksum_file})"
        ((fail_count++))
        continue
    fi

    # ダウンロードしたファイルのMD5値を計算
    calculated_md5=$(md5sum "$local_data_file" | awk '{print $1}')
    if [ -z "$calculated_md5" ]; then
        echo "    エラー: MD5値を計算できませんでした (${local_data_file})"
        ((fail_count++))
        continue
    fi

    # 値を比較
    if [ "${calculated_md5}" = "${expected_md5}" ]; then
        echo "    結果: 成功"
        ((success_count++))
    else
        echo "    結果: 失敗 (チェックサム不一致)"
        echo "      期待値: ${expected_md5}"
        echo "      計算値: ${calculated_md5}"
        ((fail_count++))
    fi
done

echo ""
echo "--- 検証結果サマリー ---"
echo "検証対象SDFファイル総数: ${total_files}"
echo "検証成功: ${success_count}"
echo "検証失敗: ${fail_count}"
if [ $missing_md5_count -gt 0 ]; then
  echo "  (うち、MD5ファイルが見つからなかったファイル数: ${missing_md5_count})"
fi
echo "--------------------------"

if [ $fail_count -gt 0 ] || [ $download_errors -gt 0 ]; then
  exit 1
else
  exit 0
fi