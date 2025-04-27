import os
import pandas as pd
import numpy as np # For array_split
from rdkit import Chem
import argparse
import math # For ceil
import time # For progress reporting (optional)


'''
# 例1: input_csvs ディレクトリのファイルを処理し、output_processed ディレクトリに保存。
#      1ファイル100万行を超える場合は分割する (デフォルト)
python 01_process_smiles.py ./input_csvs ./output_processed

# 例2: 1ファイルあたり最大10万行で分割する場合
python 01_process_smiles.py ./input_csvs ./output_processed --rows 100000
'''


def analyze_smiles(smiles):
    # ... (前回のコードと同じ) ...
    results = {
        'has_dot': False, 'num_C': 0, 'num_H': 0, 'num_O': 0, 'num_N': 0,
        'num_F': 0, 'num_Cl': 0, 'num_Br': 0, 'num_P': 0, 'num_S': 0,
        'num_Others': 0, 'has_charge': False, 'parse_error': False
    }
    if not isinstance(smiles, str) or not smiles:
        results['parse_error'] = True
        return pd.Series(results)
    if '.' in smiles: results['has_dot'] = True
    if '+' in smiles or '-' in smiles: results['has_charge'] = True
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        results['parse_error'] = True
        return pd.Series(results)
    target_elements = {'C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'P', 'S'}
    try:
        mol_with_hs = Chem.AddHs(mol)
        for atom in mol_with_hs.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in target_elements:
                results[f'num_{symbol}'] += 1
            else:
                results['num_Others'] += 1
    except Exception as e:
        print(f"Warning: Error counting elements for SMILES '{smiles}': {e}")
        results['parse_error'] = True
    return pd.Series(results)

def save_dataframe(df, base_filepath, rows_per_file=1000000):
    """
    DataFrameを指定されたパスに保存する。行数が閾値を超える場合は分割して保存する。

    Args:
        df (pd.DataFrame): 保存するDataFrame。
        base_filepath (str): 保存するファイルパスのベース (例: /path/to/output_processed.csv)。
                                分割が必要な場合、このパスにサフィックスが付与される。
        rows_per_file (int): 1ファイルあたりの最大行数。
    """
    total_rows = len(df)

    if total_rows == 0:
        print("  DataFrame is empty. Skipping save.")
        return

    output_dir = os.path.dirname(base_filepath)
    base_filename_without_ext = os.path.splitext(os.path.basename(base_filepath))[0]

    if total_rows <= rows_per_file:
        # 行数が閾値以下なら、そのまま保存
        print(f"  Saving {total_rows} rows to {base_filepath}")
        try:
            df.to_csv(base_filepath, index=False, encoding='utf-8-sig')
            print(f"  Successfully saved.")
        except Exception as e:
            print(f"  Error saving file {base_filepath}: {e}")
    else:
        # 行数が閾値を超えるなら、分割して保存
        num_files = math.ceil(total_rows / rows_per_file)
        padding = max(1, len(str(num_files - 1))) # ゼロパディング桁数
        print(f"  Total rows ({total_rows}) exceed limit ({rows_per_file}). Splitting into {num_files} files...")

        try:
            # DataFrameを分割
            split_dfs = np.array_split(df, num_files)
            del df # 元の大きなDataFrameをメモリから解放

            for i, df_part in enumerate(split_dfs):
                # 分割後のファイル名を生成 (例: base_processed_000.csv)
                part_filename = f"{base_filename_without_ext}_{i:0{padding}d}.csv"
                part_filepath = os.path.join(output_dir, part_filename)
                print(f"    Saving part {i+1}/{num_files} to {part_filepath} ({len(df_part)} rows)...")
                # CSV保存
                df_part.to_csv(part_filepath, index=False, encoding='utf-8-sig')

            print(f"  Successfully split and saved into {num_files} files.")
            del split_dfs # 分割後のリストも解放

        except Exception as e:
            print(f"  Error during file splitting or saving for base {base_filename_without_ext}: {e}")


def process_csv_files(input_dir, output_dir, rows_per_output_file=1000000):
    """
    指定されたディレクトリ内のCSVファイルを処理し、SMILES解析結果を追加して
    新しいファイルとして出力ディレクトリに保存する関数。
    出力ファイルが指定行数を超える場合は自動的に分割される。

    Args:
        input_dir (str): 入力CSVファイルが含まれるディレクトリのパス。
        output_dir (str): 処理結果のCSVファイルを保存するディレクトリのパス。
        rows_per_output_file (int): 出力CSVファイル1つあたりの最大行数。
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found or is not a directory.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}': {e}")
            return

    print(f"Starting processing files in: {os.path.abspath(input_dir)}")
    print(f"Outputting results to: {os.path.abspath(output_dir)}")
    print(f"Max rows per output file set to: {rows_per_output_file}")

    # 進捗表示用の準備
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    total_files = len(all_files)
    start_time_total = time.time()
    processed_count = 0
    skipped_count = 0

    # os.listdirではなくglobを使うとフルパスで取得できる
    # search_pattern = os.path.join(input_dir, "*.csv")
    # all_files_paths = glob.glob(search_pattern)
    # total_files = len(all_files_paths)

    for i, filename in enumerate(all_files):
        # for i, input_filepath in enumerate(all_files_paths):
        # filename = os.path.basename(input_filepath)

        input_filepath = os.path.join(input_dir, filename)
        # 出力ファイル名のベース（拡張子なし）
        output_filename_base = os.path.splitext(filename)[0] + "_processed"
        # 分割されない場合のフルパス（save_dataframe関数で使用）
        output_base_filepath = os.path.join(output_dir, output_filename_base + ".csv")

        print(f"\n--- Processing file {i+1}/{total_files}: {filename} ---")
        start_time_file = time.time()

        try:
            # CSVファイルを読み込む (エンコーディング指定とエラー処理)
            df = None
            try:
                # チャンクで読むオプションも検討可能だが、まずはシンプルに全体読み込み
                df = pd.read_csv(input_filepath)
            except UnicodeDecodeError:
                try:
                    print("  Trying encoding: shift_jis...")
                    df = pd.read_csv(input_filepath, encoding='shift_jis')
                except Exception as e_enc:
                    print(f"  Error reading file {filename} with multiple encodings: {e_enc}")
                    skipped_count += 1
                    continue
            except pd.errors.EmptyDataError:
                print(f"  Skipping empty file: {filename}")
                skipped_count += 1
                continue
            except FileNotFoundError:
                 print(f"  Error: File not found: {filename}")
                 skipped_count += 1
                 continue
            except Exception as e_read:
                 print(f"  An unexpected error occurred while reading {filename}: {e_read}")
                 skipped_count += 1
                 continue

            if df is None or df.empty:
                print(f"  Skipping file {filename} as it is empty or could not be read.")
                skipped_count += 1
                continue

            # 必須列の確認 ('cid' または 'smiles') - より堅牢に
            df_columns_lower = [col.lower() for col in df.columns]
            cid_col_name = None
            smiles_col_name = None

            # 'cid' 列を探す (大文字小文字区別なし)
            for col in df.columns:
                if col.lower() == 'cid':
                    cid_col_name = col
                    break
            if cid_col_name is None and len(df.columns) > 0:
                cid_col_name = df.columns[0] # 見つからなければ最初の列と仮定
                print(f"  Warning: 'cid' column not found by name, assuming first column ('{cid_col_name}') is cid.")

            # 'smiles' 列を探す (大文字小文字区別なし)
            for col in df.columns:
                if col.lower() == 'smiles':
                    smiles_col_name = col
                    break
            if smiles_col_name is None and len(df.columns) > 1:
                smiles_col_name = df.columns[1] # 見つからなければ2番目の列と仮定
                print(f"  Warning: 'smiles' column not found by name, assuming second column ('{smiles_col_name}') is smiles.")

            # 最終チェック
            if cid_col_name is None or smiles_col_name is None or smiles_col_name not in df.columns:
                print(f"  Skipping {filename}: Could not identify required 'cid' or 'smiles' columns.")
                skipped_count += 1
                continue

            print(f"  Identified columns - CID: '{cid_col_name}', SMILES: '{smiles_col_name}'")
            print(f"  Input data shape: {df.shape}")

            # SMILES列に対して解析関数を適用
            # df.apply は遅いことがあるので、リスト内包表記の方が速い場合がある
            print(f"  Applying SMILES analysis...")
            start_time_analysis = time.time()
            try:
                # applyを使う場合
                # analysis_results = df[smiles_col_name].apply(analyze_smiles)

                # リスト内包表記を使う場合 (速いことが多い)
                smiles_list = df[smiles_col_name].tolist()
                results_list = [analyze_smiles(s) for s in smiles_list]
                analysis_results_df = pd.DataFrame(results_list)
                # 元のインデックスに合わせる
                analysis_results_df.index = df.index

            except Exception as e_apply:
                print(f"  Error applying analysis function to smiles column: {e_apply}")
                skipped_count += 1
                continue
            analysis_time = time.time() - start_time_analysis
            print(f"  SMILES analysis completed in {analysis_time:.2f} seconds.")

            # 元のDataFrameと解析結果を結合
            df_processed = pd.concat([df, analysis_results_df], axis=1)
            print(f"  Processed data shape: {df_processed.shape}")

            # ----- 結果の保存 (分割処理を含む) -----
            print(f"  Saving processed data...")
            save_dataframe(df_processed, output_base_filepath, rows_per_file=rows_per_output_file)
            # ------------------------------------

            processed_count += 1
            # メモリ解放（任意）
            del df, analysis_results_df, df_processed

        except Exception as e:
            print(f"  An unexpected error occurred while processing {filename}: {e}")
            skipped_count += 1

        file_time = time.time() - start_time_file
        print(f"--- File processing time: {file_time:.2f} seconds ---")


    # --- 全体処理完了 ---
    total_time = time.time() - start_time_total
    print(f"\n==============================================")
    print(f"All processing finished.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successfully processed files: {processed_count}")
    print(f"Skipped or failed files: {skipped_count}")
    print(f"==============================================")


# --- メイン処理 ---
if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(
        description="Process CSV files containing SMILES strings, add molecular analysis columns, and save results, splitting large files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        "input_dir",
        help="Path to the directory containing input CSV files."
        )
    parser.add_argument(
        "output_dir",
        help="Path to the directory where processed CSV files will be saved."
        )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000000, # デフォルト100万行
        help="Maximum number of rows per single output CSV file. Files exceeding this limit will be split."
        )
    # チャンク読み込みオプションは一旦削除 (applyやリスト内包との組み合わせが複雑になるため)
    # 必要であれば後で追加検討

    # 引数を解析
    args = parser.parse_args()

    # 処理を実行
    process_csv_files(args.input_dir, args.output_dir, args.rows)