# settings.py
import os

# --- 定数定義 ---
# 環境変数で上書き可能にして再現パッケージから制御しやすくする
RANDOM_STATE = int(os.getenv('VULJIT_RANDOM_STATE', '42'))
N_SPLITS_K = int(os.getenv('VULJIT_N_SPLITS_K', '10'))  # 10分割交差検証
# TODO: 30回が本番
N_REPETITIONS = int(os.getenv('VULJIT_N_REPETITIONS', '30'))  # 評価サイクルの繰り返し回数

# 予測確率を二値に変換する際の既定閾値（CSV出力用）
# 環境変数 VULJIT_LABEL_THRESHOLD で上書き可能
PREDICTION_LABEL_THRESHOLD = float(os.getenv('VULJIT_LABEL_THRESHOLD', '0.5'))

# sample数のしきい値
# 少数クラスのサンプル数がこの値未満の場合、プロジェクトをスキップする
MIN_SAMPLES_THRESHOLD = 10


# ランダムサーチの設定
# 内部CV分割数
RANDOM_SEARCH_CV = 3
# 試行回数　
RANDOM_SEARCH_N_ITER = 20
SCORING = {
    "PR_AUC": "average_precision",
    "MCC": "matthews_corrcoef",
    "F1": "f1",
    "Precision": "precision",
    "Recall": "recall",
    "AUC": "roc_auc",
}




# --- 評価方法設定 ---
# 'stratified_k_fold' または 'time_series' を選択
EVALUATION_METHOD = os.getenv('VULJIT_EVAL_METHOD', 'time_series')  # 'stratified_k_fold' or 'time_series'
# --- 時系列交差検証の設定 ---
# データセットを分割するチャンク（固まり）の数
N_SPLITS_TIMESERIES = int(os.getenv('VULJIT_N_SPLITS_TIMESERIES', '10'))
# Trueにすると直前のFoldのみで学習（スライディングウィンドウ）、Falseにすると累積データで学習
USE_ONLY_RECENT_FOR_TRAINING = os.getenv('VULJIT_USE_ONLY_RECENT', 'false').lower() in ('1', 'true', 'yes')
USE_HYPERPARAM_OPTIMIZATION = os.getenv('VULJIT_USE_HPO', 'false').lower() in ('1', 'true', 'yes')

# 単純ホールドアウト分割（時系列順を保持）
# 先頭からこの割合を学習、残りをテストに割り当てる（0.05〜0.95にクリップ）
SIMPLE_SPLIT_TRAIN_RATIO = 0.5


# データサンプリング手法
# 'random_under' または 'smote' を選択
SAMPLING_METHOD = 'random_under'
# ランダムアンダーサンプリングの戦略
RANDOM_UNDER_SAMPLING_STRATEGY = 'majority'
# --- モデル 定義 ---
# 使用するモデルのリスト 'random_forest', 'xgboost', 'random'
SELECTED_MODEL = os.getenv('VULJIT_MODEL', 'random_forest')  # 'random_forest' or 'xgboost' or 'random'
# 使用可能なモデルのリスト
AVAILABLE_MODELS = ['random_forest', 'xgboost','random']
# random予測の戦略
RANDOM_BASELINE_STRATEGY = 'stratified'


# --- パス定義 ---
# このパスはご自身の環境に合わせて変更してください
# 環境変数で上書き: VULJIT_BASE_DATA_DIR, VULJIT_RESULTS_DIR
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, '..'))
BASE_DATA_DIRECTORY = os.getenv('VULJIT_BASE_DATA_DIR', os.path.join(_repo_root, 'data'))

# 既定の結果出力ルート（モデル共通の親ディレクトリ）
_RESULTS_PARENT_DIR = os.getenv('VULJIT_RESULTS_DIR', os.path.join(_repo_root, 'outputs', 'results'))

# モデルごとに保存ディレクトリを分ける（例: outputs/results/random_forest/...）
_MODEL_NAME_SAFE = (SELECTED_MODEL or 'model').strip().lower().replace(' ', '_')
RESULTS_BASE_DIRECTORY = os.path.join(_RESULTS_PARENT_DIR, _MODEL_NAME_SAFE)

# === 出力先: モデルとログ ===
# モデルの保存先（パスは任意に変更可）
MODEL_OUTPUT_DIRECTORY = RESULTS_BASE_DIRECTORY
# ハイパーパラ探索結果やその他ログの保存先
LOGS_DIRECTORY = os.path.join(RESULTS_BASE_DIRECTORY, 'logs') if 'os' in globals() else RESULTS_BASE_DIRECTORY + '/logs'

# 保存制御
SAVE_BEST_MODEL = False
SAVE_HYPERPARAM_RESULTS = False

# --- 特徴量セット定義 ---

# kemei特徴量
KAMEI_FEATURES = [
    "subsystems_changed", "directories_changed", "files_changed", "total_lines_changed", "lines_added", "lines_deleted", 
    "total_prev_loc", "is_bug_fix", "past_bug_fixes", "entropy", "ndev", "age", "nuc", "exp", "rexp", "sexp",
        ]

# VCCfinder特徴量
VCCFINDER_FEATURES = [
    "VCC_s1_nb_added_sizeof",
    "VCC_s2_nb_removed_sizeof",
    "VCC_s3_diff_sizeof",
    "VCC_s4_sum_sizeof",
    "VCC_s5_nb_added_continue",
    "VCC_s6_nb_removed_continue",
    "VCC_s7_nb_added_break",
    "VCC_s8_nb_removed_break",
    "VCC_s9_nb_added_INTMAX",
    "VCC_s10_nb_removed_INTMAX",
    "VCC_s11_nb_added_goto",
    "VCC_s12_nb_removed_goto",
    "VCC_s13_nb_added_define",
    "VCC_s14_nb_removed_define",
    "VCC_s15_nb_added_struct",
    "VCC_s16_nb_removed_struct",
    "VCC_s17_diff_struct",
    "VCC_s18_sum_struct",
    "VCC_s19_nb_added_offset",
    "VCC_s20_nb_removed_offset",
    "VCC_s21_nb_added_void",
    "VCC_s22_nb_removed_void",
    "VCC_s23_diff_void",
    "VCC_s24_sum_void",
    "VCC_f1_sum_file_change",
    "VCC_f2_nb_added_loop",
    "VCC_f3_nb_removed_loop",
    "VCC_f4_diff_loop",
    "VCC_f5_sum_loop",
    "VCC_f6_nb_added_if",
    "VCC_f7_nb_removed_if",
    "VCC_f8_diff_if",
    "VCC_f9_sum_if",
    "VCC_f10_nb_added_line",
    "VCC_f11_nb_removed_line",
    "VCC_f12_diff_line",
    "VCC_f13_sum_line",
    "VCC_f14_nb_added_paren",
    "VCC_f15_nb_removed_paren",
    "VCC_f16_diff_paren",
    "VCC_f17_sum_paren",
    "VCC_f18_nb_added_bool",
    "VCC_f19_nb_removed_bool",
    "VCC_f20_diff_bool",
    "VCC_f21_sum_bool",
    "VCC_f22_nb_added_assignement",
    "VCC_f23_nb_removed_assignement",
    "VCC_f24_diff_assignement",
    "VCC_f25_sum_assignement",
    "VCC_f26_nb_added_function",
    "VCC_f27_nb_removed_function",
    "VCC_f28_diff_function",
    "VCC_f29_sum_function",
    "VCC_w1",
    "VCC_w2",
    "VCC_w3",
    "VCC_w4",
    "VCC_w5",
    "VCC_w6",
    "VCC_w7",
    "VCC_w8",
    "VCC_w9",
    "VCC_w10",
]





# --- カバレッジ特徴量 ---




# === プロジェクトベース ===

# 4. プロジェクトベース・コミット単位 (パーセンテージ)
PROJECT_COMMIT_PERCENT_FEATURES = [
]

# 5. プロジェクトベース・プロジェクト全体 (パーセンテージ)
PROJECT_TOTAL_PERCENT_FEATURES = [
    "project_total_function_percent",
    "project_total_line_percent",
    "project_total_region_percent",
    "project_total_branch_percent",
    "project_total_instantiation_percent",
    "patch_coverage_recalculated",
    "project_total_function_percent_delta",
    "project_total_line_percent_delta",
    "project_total_region_percent_delta",
    "project_total_branch_percent_delta",
    "project_total_instantiation_percent_delta",
    "patch_coverage_recalculated_delta",
    # 
    # "daily_commit_count",
]

# 6. プロジェクトベース・計算用特徴量 (covered, count)
PROJECT_CALCULATION_FEATURES = [
    # プロジェクト全体
    "project_total_function_covered", "project_total_function_count",
    "project_total_line_covered", "project_total_line_count",
    "project_total_region_covered", "project_total_region_count",
    "project_total_branch_covered", "project_total_branch_count",
    "project_total_instruction_covered", "project_total_instruction_count",
    # パッチカバレッジ
    "covered_added_lines","total_added_lines",
]

# プロジェクトベース・カバレッジ
PROJECT_ALL_PERCENT_FEATURES = (
    PROJECT_COMMIT_PERCENT_FEATURES +
    PROJECT_TOTAL_PERCENT_FEATURES
)



# === 統合リスト ===

# 7. 全てのパーセンテージ特徴量
ALL_PERCENT_FEATURES = (
    PROJECT_COMMIT_PERCENT_FEATURES +
    PROJECT_TOTAL_PERCENT_FEATURES
)

# 8. 全ての計算用特徴量
ALL_CALCULATION_FEATURES = (
    PROJECT_CALCULATION_FEATURES
)
