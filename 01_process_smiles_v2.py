import os
import pandas as pd
import numpy as np
from rdkit import Chem
import argparse
import math
import time
import multiprocessing # multiprocessingをインポート
# from tqdm import tqdm # オプション: 進捗表示を改善したい場合

'''
# 例1: input_csvs ディレクトリのファイルを処理し、output_processed ディレクトリに保存。
#      1ファイル100万行を超える場合は分割する (デフォルト)。ワーカー数は自動設定。
python 01_process_smiles_mp.py ./input_csvs ./output_processed

# 例2: 1ファイルあたり最大10万行で分割、ワーカー数を4に指定する場合
python 01_process_smiles_mp.py ./input_csvs ./output_processed --rows 100000 --workers 4
'''

def analyze_smiles(smiles):
    """SMILES文字列を解析して分子情報を返す関数 (変更なし)"""
    results = {
        'has_dot': False, 'num_C': 0, 'num_H': 0, 'num_O': 0, 'num_N': 0,
        'num_F': 0, 'num_Cl': 0, 'num_Br': 0, 'num_P': 0, 'num_S': 0,
        'num_Others': 0, 'has_charge': False, 'parse_error': False
    }
    if not isinstance(smiles, str) or not smiles:
        results['parse_error'] = True
        # multiprocessingで使う場合、純粋な辞書やタプルを返す方が安定することがある
        # return results
        return pd.Series(results) # 今回はSeriesのままでも大丈夫そう

    # RDKitは'.'を含むSMILESを解釈できない場合があるため、先にチェック
    if '.' in smiles: results['has_dot'] = True
    # RDKitは電荷を含むSMILESも解釈に失敗することがあるため、先にチェック
    if '+' in smiles or '-' in smiles: results['has_charge'] = True

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        results['parse_error'] = True
        # return results
        return pd.Series(results)

    target_elements = {'C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'P', 'S'}
    try:
        # 水素原子を追加してカウントする
        mol_with_hs = Chem.AddHs(mol)
        for atom in mol_with_hs.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in target_elements:
                # `f'num_{symbol}'` のキーが存在するか確認する方がより安全だが、
                # results辞書に全てのキーが初期化されているので大丈夫
                results[f'num_{symbol}'] += 1
            else:
                results['num_Others'] += 1
    except Exception as e:
        # 並列処理中はprintが混ざると見づらいので、エラー発生の事実だけ記録
        # 詳細なエラーログが必要な場合は、loggingモジュールなどを検討
        # print(f"Warning: Error counting elements for SMILES '{smiles}': {e}")
        results['parse_error'] = True # AddHsやGetAtomsでのエラーもパースエラー扱い

    # return results
    return pd.Series(results)


def save_dataframe(df, base_filepath, rows_per_file=1000000):
    """DataFrameを指定されたパスに保存する関数 (変更なし)"""
    total_rows = len(df)

    if total_rows == 0:
        print("   DataFrame is empty. Skipping save.")
        return

    output_dir = os.path.dirname(base_filepath)
    base_filename_without_ext = os.path.splitext(os.path.basename(base_filepath))[0]

    if total_rows <= rows_per_file:
        print(f"   Saving {total_rows} rows to {base_filepath}")
        try:
            df.to_csv(base_filepath, index=False, encoding='utf-8-sig')
            print(f"   Successfully saved.")
        except Exception as e:
            print(f"   Error saving file {base_filepath}: {e}")
    else:
        num_files = math.ceil(total_rows / rows_per_file)
        padding = max(1, len(str(num_files - 1)))
        print(f"   Total rows ({total_rows}) exceed limit ({rows_per_file}). Splitting into {num_files} files...")

        try:
            split_dfs = np.array_split(df, num_files)
            del df # メモリ解放

            for i, df_part in enumerate(split_dfs):
                part_filename = f"{base_filename_without_ext}_{i:0{padding}d}.csv"
                part_filepath = os.path.join(output_dir, part_filename)
                print(f"     Saving part {i+1}/{num_files} to {part_filepath} ({len(df_part)} rows)...")
                df_part.to_csv(part_filepath, index=False, encoding='utf-8-sig')

            print(f"   Successfully split and saved into {num_files} files.")
            del split_dfs

        except Exception as e:
            print(f"   Error during file splitting or saving for base {base_filename_without_ext}: {e}")

def process_csv_files(input_dir, output_dir, rows_per_output_file=1000000, num_workers=None):
    """
    指定されたディレクトリ内のCSVファイルを処理し、SMILES解析結果を追加して
    新しいファイルとして出力ディレクトリに保存する関数（並列処理版）。

    Args:
        input_dir (str): 入力CSVファイルが含まれるディレクトリのパス。
        output_dir (str): 処理結果のCSVファイルを保存するディレクトリのパス。
        rows_per_output_file (int): 出力CSVファイル1つあたりの最大行数。
        num_workers (int, optional): 使用するプロセス数。Noneの場合はCPUコア数を使用。
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

    # --- 使用するプロセス数を決定 ---
    if num_workers is None:
        try:
            num_workers = os.cpu_count()
            if num_workers is None:
                 num_workers = 1
                 print("Warning: Could not detect CPU core count. Using 1 worker.")
            # マシンリソースを使いすぎないように調整する場合 (例: 1コア余裕を持たせる)
            # num_workers = max(1, os.cpu_count() - 1)
        except NotImplementedError:
            num_workers = 1
            print("Warning: Could not detect CPU core count (NotImplementedError). Using 1 worker.")
    num_workers = max(1, num_workers) # 最低1は確保

    print(f"Starting processing files in: {os.path.abspath(input_dir)}")
    print(f"Outputting results to: {os.path.abspath(output_dir)}")
    print(f"Max rows per output file set to: {rows_per_output_file}")
    print(f"Using {num_workers} worker processes for analysis.")

    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    total_files = len(all_files)
    start_time_total = time.time()
    processed_count = 0
    skipped_count = 0

    for i, filename in enumerate(all_files):
        input_filepath = os.path.join(input_dir, filename)
        output_filename_base = os.path.splitext(filename)[0] + "_processed"
        output_base_filepath = os.path.join(output_dir, output_filename_base + ".csv")

        print(f"\n--- Processing file {i+1}/{total_files}: {filename} ---")
        start_time_file = time.time()

        df = None # ループ外でも参照される可能性があるので初期化
        analysis_results_df = None
        df_processed = None

        try:
            # --- CSVファイルの読み込み (変更なし) ---
            try:
                df = pd.read_csv(input_filepath, low_memory=False) # low_memory=False はメモリを多く使うが読み込みが速いことがある
            except UnicodeDecodeError:
                try:
                    print("   Trying encoding: shift_jis...")
                    df = pd.read_csv(input_filepath, encoding='shift_jis', low_memory=False)
                except Exception as e_enc:
                    print(f"   Error reading file {filename} with multiple encodings: {e_enc}")
                    skipped_count += 1
                    continue
            except pd.errors.EmptyDataError:
                print(f"   Skipping empty file: {filename}")
                skipped_count += 1
                continue
            except FileNotFoundError:
                 print(f"   Error: File not found: {filename}")
                 skipped_count += 1
                 continue
            except Exception as e_read:
                 print(f"   An unexpected error occurred while reading {filename}: {e_read}")
                 skipped_count += 1
                 continue

            if df is None or df.empty:
                print(f"   Skipping file {filename} as it is empty or could not be read.")
                skipped_count += 1
                continue

            # --- 必須列の確認 (変更なし) ---
            df_columns_lower = [col.lower() for col in df.columns]
            cid_col_name = None
            smiles_col_name = None

            # 'cid' 列の探索 (より柔軟に)
            potential_cid_cols = [col for col in df.columns if 'cid' in col.lower()]
            if 'cid' in df_columns_lower:
                cid_col_name = df.columns[df_columns_lower.index('cid')]
            elif potential_cid_cols:
                 cid_col_name = potential_cid_cols[0]
                 print(f"   Warning: Exact 'cid' column not found, using potential match: '{cid_col_name}'")
            elif len(df.columns) > 0:
                cid_col_name = df.columns[0]
                print(f"   Warning: 'cid' column not found by name or pattern, assuming first column ('{cid_col_name}') is cid.")

            # 'smiles' 列の探索 (より柔軟に)
            potential_smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
            if 'smiles' in df_columns_lower:
                 smiles_col_name = df.columns[df_columns_lower.index('smiles')]
            elif potential_smiles_cols:
                 smiles_col_name = potential_smiles_cols[0]
                 print(f"   Warning: Exact 'smiles' column not found, using potential match: '{smiles_col_name}'")
            elif len(df.columns) > 1:
                smiles_col_name = df.columns[1]
                print(f"   Warning: 'smiles' column not found by name or pattern, assuming second column ('{smiles_col_name}') is smiles.")

            if cid_col_name is None or smiles_col_name is None or smiles_col_name not in df.columns:
                print(f"   Skipping {filename}: Could not identify required 'cid' or 'smiles' columns.")
                skipped_count += 1
                del df # 不要になったdfを解放
                continue

            print(f"   Identified columns - CID: '{cid_col_name}', SMILES: '{smiles_col_name}'")
            print(f"   Input data shape: {df.shape}")
            num_rows = len(df)

            # --- SMILES解析を並列実行 ---
            print(f"   Applying SMILES analysis using {num_workers} workers...")
            start_time_analysis = time.time()

            # SMILESリストを取得 (NaNは空文字列に置換)
            smiles_list = df[smiles_col_name].fillna('').tolist()

            analysis_results_df = None # 初期化
            if smiles_list: # 処理すべきSMILESがある場合のみ実行
                try:
                    # multiprocessing.Pool を使用
                    # map は順序保証、imap_unordered は順序保証なしだがメモリ効率が良い場合がある
                    # 元のDFと結合するので順序保証がある map が扱いやすい
                    # chunksize を適切に設定すると効率が上がることがある
                    # あまりに小さいとオーバーヘッドが大きくなり、大きすぎると負荷が偏る可能性
                    chunk_size = max(1, math.ceil(num_rows / num_workers / 4)) # 経験則的な値
                    if num_rows < num_workers * 2: # データが非常に少ない場合はchunksizeを調整
                        chunk_size = 1

                    print(f"     Using chunksize: {chunk_size}")

                    with multiprocessing.Pool(processes=num_workers) as pool:
                        # analyze_smiles が pd.Series を返すので、それのリストが返る
                        results_series_list = pool.map(analyze_smiles, smiles_list, chunksize=chunk_size)
                        # オプション: tqdmで進捗表示する場合
                        # results_series_list = list(tqdm(pool.imap(analyze_smiles, smiles_list, chunksize=chunk_size), total=num_rows, desc="Analyzing SMILES"))


                    # 結果 (Seriesのリスト) をDataFrameに変換
                    if results_series_list:
                        # concatがメモリを大量に消費する場合があるので注意
                        # 必要なら結果を一旦ディスクに書き出すなどの工夫も検討
                        analysis_results_df = pd.concat(results_series_list, axis=1).T # Transposeで正しい形に
                        analysis_results_df.index = df.index # 元のインデックスに合わせる
                    else:
                         # 結果リストが空の場合（通常は起こらないはずだが念のため）
                         print("     Warning: Analysis returned an empty list.")
                         analysis_results_df = pd.DataFrame(index=df.index) # 空のDFを作成


                except Exception as e_mp:
                    print(f"   Error during parallel analysis: {e_mp}")
                    # エラー発生時はスキップ
                    skipped_count += 1
                    del df, smiles_list # メモリ解放
                    continue # 次のファイルへ
            else:
                # SMILESリストが空の場合
                print("   SMILES list is empty. Skipping analysis.")
                # 空の解析結果DataFrameを作成（カラム名はanalyze_smilesの返すキーに合わせる）
                analysis_columns = list(analyze_smiles("C").keys()) # ダミー実行でカラム名取得
                analysis_results_df = pd.DataFrame(columns=analysis_columns, index=df.index, dtype=object) # dtypeを指定するとより良い


            analysis_time = time.time() - start_time_analysis
            print(f"   SMILES analysis completed in {analysis_time:.2f} seconds.")

            # --- 結果の結合と保存 ---
            if analysis_results_df is not None:
                # dfとanalysis_results_dfの行数が一致しているか確認（念のため）
                if len(df) == len(analysis_results_df):
                    df_processed = pd.concat([df, analysis_results_df], axis=1)
                    print(f"   Processed data shape: {df_processed.shape}")

                    print(f"   Saving processed data...")
                    save_dataframe(df_processed, output_base_filepath, rows_per_file=rows_per_output_file)
                    processed_count += 1
                else:
                    print(f"   Error: Mismatch in row count between original data ({len(df)}) and analysis results ({len(analysis_results_df)}). Skipping save.")
                    skipped_count += 1

            else:
                 # analysis_results_df が None のままの場合（並列処理エラーなどで作成されなかった場合）
                 print(f"   Skipping save for {filename} due to analysis result being None.")
                 skipped_count += 1

            # --- メモリ解放 ---
            del df, smiles_list, analysis_results_df, df_processed

        except Exception as e:
            print(f"   An unexpected error occurred during the main processing loop for {filename}: {e}")
            import traceback
            traceback.print_exc() # 詳細なエラー箇所を出力
            skipped_count += 1
            # エラー時にオブジェクトが残っている可能性があるので解放トライ
            try: del df
            except NameError: pass
            try: del smiles_list
            except NameError: pass
            try: del analysis_results_df
            except NameError: pass
            try: del df_processed
            except NameError: pass


        file_time = time.time() - start_time_file
        print(f"--- File processing time: {file_time:.2f} seconds ---")
        # Optional: ガベージコレクションを強制する (効果は限定的かもしれない)
        # import gc
        # gc.collect()


    # --- 全体処理完了 ---
    total_time = time.time() - start_time_total
    print(f"\n==============================================")
    print(f"All processing finished.")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Successfully processed files: {processed_count}")
    print(f"Skipped or failed files: {skipped_count}")
    print(f"==============================================")


# --- メイン処理 (引数に --workers を追加) ---
if __name__ == "__main__":
    # Windowsでmultiprocessingを使う場合のおまじない
    # (スクリプトが直接実行された時のみ以下のコードが動くようにする)
    multiprocessing.freeze_support() # Windows環境で実行ファイル化する際などに必要

    parser = argparse.ArgumentParser(
        description="Process CSV files containing SMILES strings using multiprocessing, add molecular analysis columns, and save results, splitting large files.",
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
        default=1000000,
        help="Maximum number of rows per single output CSV file. Files exceeding this limit will be split."
        )
    parser.add_argument(
        "--workers",
        type=int,
        default=None, # デフォルトNoneで、関数内でコア数を自動設定
        help="Number of worker processes to use for SMILES analysis. Defaults to the number of CPU cores available."
        )

    args = parser.parse_args()

    # 処理を実行
    process_csv_files(args.input_dir, args.output_dir, args.rows, args.workers)