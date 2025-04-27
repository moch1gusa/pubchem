import os
import pandas as pd
import numpy as np # DataFrame分割のため (pip install numpy が必要)
import glob
import argparse
import time
import math # 天井関数 (ceil) のため

'''
# 例: processed_csvディレクトリのファイルをフィルタリングし、
# resultsディレクトリに 'filtered_part_XXX.csv' という名前で
# 1ファイル最大10万行で分割保存（チャンク処理有効）
python split_filtered_data.py ./processed_csv ./results --prefix filtered_part_ --rows 100000 --chunk

# 例: デフォルト設定（1ファイル100万行）で分割
python split_filtered_data.py ./processed_csv ./results --chunk
'''

# --- ここに抽出条件を定義 ---
def filter_dataframe_by_conditions(df):
    """
    DataFrameを受け取り、指定した条件でフィルタリングして返す関数。
    この関数内の条件式を編集して、目的の抽出条件を設定してください。

    Args:
        df (pd.DataFrame): フィルタリング対象のDataFrame。

    Returns:
        pd.DataFrame: フィルタリング後のDataFrame。条件に合う行がない場合は空のDataFrame。
    """
    try:
        # --- 抽出条件の例 ---

        condition = (df['has_dot'] == False) & (df['num_C'] >= 8) & (df['num_C'] <= 18) & (df['num_S'] == 0)& (df['num_P'] == 0) & (df['num_Br'] == 0)& (df['num_Cl'] == 0) & (df['num_Br'] == 0) & (df['num_Others'] == 0) &  (df['parse_error'] == False) 


        # --- ここまで ---

        # 指定した条件でフィルタリング
        # フィルタリング前に列が存在するか確認するとより安全
        required_columns = [] # 条件式で使っている列名をリストアップ (例: ['has_dot', 'num_C'])
        # if not all(col in df.columns for col in required_columns):
        #    print(f"  Warning: One or more required columns for filtering not found. Skipping.")
        #    return pd.DataFrame()

        filtered_df = df[condition].copy() # .copy() をつけてSettingWithCopyWarningを抑制
        return filtered_df

    except KeyError as e:
        print(f"  Warning: Filtering condition refers to a non-existent column: {e}. Returning empty DataFrame.")
        return pd.DataFrame() # 条件に必要な列がない場合は空を返す
    except Exception as e:
        print(f"  Warning: An error occurred during filtering: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

def combine_filter_and_split(processed_dir, output_dir, output_prefix, rows_per_file, use_chunking=False, chunk_size=50000):
    """
    処理済みCSVを読み込み、フィルタリングし、結合後の総行数を表示し、
    指定された行数で複数のCSVファイルに分割して保存する。

    Args:
        processed_dir (str): 処理済みCSVファイル (*_processed.csv) があるディレクトリ。
        output_dir (str): 分割されたCSVファイルを保存するディレクトリ。
        output_prefix (str): 分割されたCSVファイル名の接頭辞 (例: 'output_part_')。
        rows_per_file (int): 分割後の1ファイルあたりの最大行数。
        use_chunking (bool): チャンク処理を使用するかどうか。
        chunk_size (int): チャンク処理を使用する場合の1チャンクあたりの行数。
    """
    start_time = time.time()

    if not os.path.isdir(processed_dir):
        print(f"Error: Input directory '{processed_dir}' not found or is not a directory.")
        return

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}': {e}")
            return

    # ディレクトリ内の *_processed.csv ファイルを検索
    search_pattern = os.path.join(processed_dir, "*_processed.csv")
    processed_files = sorted(glob.glob(search_pattern)) # ファイル順序を固定するためソート

    if not processed_files:
        print(f"No '*_processed.csv' files found in '{processed_dir}'.")
        return

    print(f"Found {len(processed_files)} processed files in '{processed_dir}'.")
    print(f"Filtering criteria are defined in the 'filter_dataframe_by_conditions' function.")
    if use_chunking:
        print(f"Using chunking with chunk size: {chunk_size} rows.")
    print(f"Output split files will be saved to: {output_dir}")
    print(f"Output file prefix: '{output_prefix}', Max rows per file: {rows_per_file}")

    # フィルタリング後のDataFrame（またはチャンク）を格納するリスト
    list_of_filtered_dfs = []
    total_rows_processed = 0
    current_total_matching_rows = 0 # 結合前の合計行数

    # 各ファイルを処理
    for i, filepath in enumerate(processed_files):
        filename = os.path.basename(filepath)
        print(f"\nProcessing file {i+1}/{len(processed_files)}: {filename} ...")
        rows_processed_in_file = 0
        rows_matching_in_file = 0
        temp_chunks_for_file = [] # このファイルから得られたフィルタリング済みチャンク

        try:
            # チャンク処理を使うか、ファイル全体を読むか
            read_iterator = pd.read_csv(filepath, chunksize=chunk_size) if use_chunking else [pd.read_csv(filepath)]

            # チャンクまたはファイル全体を処理
            for chunk_num, chunk_or_df in enumerate(read_iterator):
                rows_in_chunk = len(chunk_or_df)
                rows_processed_in_file += rows_in_chunk
                if rows_in_chunk == 0: continue # 空のチャンク/DFはスキップ

                # フィルタリングを実行
                filtered_chunk = filter_dataframe_by_conditions(chunk_or_df)

                if not filtered_chunk.empty:
                    num_matching = len(filtered_chunk)
                    rows_matching_in_file += num_matching
                    temp_chunks_for_file.append(filtered_chunk) # フィルタリング後を一時リストへ
                    # (任意) 詳細な進捗表示
                    # if use_chunking:
                    #     print(f"  Chunk {chunk_num+1}: Processed {rows_in_chunk} rows, found {num_matching} matches.", end='\r')

            # (任意) 詳細な進捗表示の改行
            # if use_chunking and rows_processed_in_file > 0: print()

            # ファイル内のフィルタリング済みチャンクを（もしあれば）結合してメインリストへ
            if temp_chunks_for_file:
                # メモリを節約するため、ファイルごとに結合してからリストに追加
                file_filtered_df = pd.concat(temp_chunks_for_file, ignore_index=True)
                list_of_filtered_dfs.append(file_filtered_df)
                # 結合前の総行数を更新
                current_total_matching_rows += len(file_filtered_df)
                # メモリ解放のため一時リストをクリア
                del temp_chunks_for_file, file_filtered_df

            print(f"  Finished file. Processed {rows_processed_in_file} rows, found {rows_matching_in_file} total matches in this file.")
            total_rows_processed += rows_processed_in_file

        except pd.errors.EmptyDataError:
            print(f"  Skipping empty file: {filename}")
        except FileNotFoundError:
             print(f"  Error: File not found: {filename}")
        except Exception as e:
            print(f"  An unexpected error occurred while processing {filename}: {e}")

    # --- フィルタリング完了後、結合前の総行数を表示 ---
    print("\n" + "="*60)
    print(f"Filtering completed across all files.")
    print(f"Total rows matching the criteria (before final concat): {current_total_matching_rows}")
    print(f"Total rows processed across all files: {total_rows_processed}")
    print("="*60)

    if current_total_matching_rows == 0:
        print("\nNo data matched the filter criteria. No output files will be generated.")
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds.")
        return

    # --- 最終的な結合処理 ---
    print("\nConcatenating filtered data...")
    final_df = None # 初期化
    try:
        # list_of_filtered_dfs に格納された DataFrame を結合
        if list_of_filtered_dfs:
             final_df = pd.concat(list_of_filtered_dfs, ignore_index=True)
             # 結合後、メモリを解放
             del list_of_filtered_dfs
             print(f"Final combined dataframe shape: {final_df.shape}")
             # 念のため行数を確認
             if len(final_df) != current_total_matching_rows:
                 print(f"Warning: Row count mismatch after concat. Expected {current_total_matching_rows}, got {len(final_df)}.")
        else:
            # このケースは基本的には発生しないはず (上でチェックしているため)
            print("Warning: No dataframes to concatenate, though total matching rows > 0 was reported.")
            return

    except Exception as e:
        print(f"Error during final concatenation: {e}")
        return

    # --- ファイル分割処理 ---
    if final_df is not None and not final_df.empty:
        print(f"\nSplitting the final dataframe into files with max {rows_per_file} rows each...")
        try:
            total_rows = len(final_df)
            # 分割数を計算
            num_files = math.ceil(total_rows / rows_per_file)
            # ゼロパディングの桁数を計算 (例: 9ファイルなら1桁, 10->2桁, 100->3桁)
            # max(1, ...) はファイル数が1の場合にlen('')=0になるのを防ぐ
            padding = max(1, len(str(num_files - 1)))

            print(f"Total rows: {total_rows}. Target rows per file: {rows_per_file}. Estimated number of files: {num_files}")

            # np.array_split を使って DataFrame をほぼ均等に分割
            # indices_or_sections に分割数を指定する
            split_dfs = np.array_split(final_df, num_files)

            # 分割後のメモリ解放のため元データを削除
            del final_df

            # 分割された DataFrame を個別の CSV ファイルとして保存
            for i, df_part in enumerate(split_dfs):
                if df_part.empty: # 空のDataFrameはスキップ
                    continue
                # ファイル名を生成 (例: output_prefix_000.csv)
                part_filename = f"{output_prefix}{i:0{padding}d}.csv"
                part_filepath = os.path.join(output_dir, part_filename)
                print(f"  Saving {part_filepath} ({len(df_part)} rows)...")
                # CSV保存
                df_part.to_csv(part_filepath, index=False, encoding='utf-8-sig')

            print(f"\nSuccessfully split data into {num_files} files in directory: {output_dir}")

        except Exception as e:
            print(f"Error during file splitting or saving: {e}")
    else:
        print("\nFinal dataframe is empty, skipping splitting.")


    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds.")

# --- メイン処理 ---
if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(
        description="Combine processed CSVs, filter based on criteria, show total matches, and split into multiple CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # デフォルト値をヘルプに表示
    )
    parser.add_argument(
        "processed_dir",
        help="Directory containing the '*_processed.csv' files generated by the previous step."
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the split output CSV files will be saved."
    )
    parser.add_argument(
        "--prefix",
        default="split_data_",
        help="Prefix for the output split file names."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000000, # デフォルトを100万行に設定
        help="Maximum number of rows per output split file."
    )
    parser.add_argument(
        "--chunk",
        action="store_true", # オプションが存在すればTrueになる
        help="Use chunking for reading input CSV files (recommended for large files to save memory)."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Number of rows per chunk when --chunk option is used."
    )

    # 引数を解析
    args = parser.parse_args()

    # 抽出条件の確認を促すメッセージ
    print("="*60)
    print("IMPORTANT: Review the filtering logic within the 'filter_dataframe_by_conditions' function in the script.")
    print("The script will proceed with the currently defined conditions.")
    print("="*60)
    # input("Press Enter to continue or Ctrl+C to cancel and edit the script...") # 必要であれば確認

    # 処理を実行
    combine_filter_and_split(
        args.processed_dir,
        args.output_dir,
        args.prefix,
        args.rows,
        use_chunking=args.chunk,
        chunk_size=args.chunksize
    )