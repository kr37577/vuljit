# `cli/additional_builds_cli.py` Flowcharts

各関数の処理フローを Mermaid 形式でまとめました。Mermaid 対応エディタやビューアに貼り付ければ図として確認できます。

## `_ensure_directory`
```mermaid
graph TD
    start(["Start"]) --> makeDirs["os.makedirs(path, exist_ok=True)"]
    makeDirs --> returnPath(["Return path"])
```

## `_load_detection_table`
```mermaid
graph TD
    start(["Start"]) --> readCsv["pd.read_csv(path)"]
    readCsv --> chooseProject{"Has column 'project'?"}
    chooseProject -- Yes --> useProject["project_col = 'project'"]
    chooseProject -- No --> usePackage["project_col = 'package_name'"]
    useProject --> normalize
    usePackage --> normalize
    normalize["Trim project names"] --> dropEmpty["Filter empty project rows"]
    dropEmpty --> hasDays{"Has column detection_time_days?"}
    hasDays -- Yes --> toNumeric["Convert detection_time_days to numeric"]
    hasDays -- No --> skipConvert["Skip conversion"]
    toNumeric --> dropNa
    skipConvert --> dropNa
    dropNa["Drop rows missing project or detection_time_days"] --> returnDf(["Return DataFrame"])
```

## `_load_build_counts`
```mermaid
graph TD
    start(["Start"]) --> readCsv["pd.read_csv(path)"]
    readCsv --> normalize["Trim project names"]
    normalize --> dropEmpty["Filter empty project rows"]
    dropEmpty --> toNumeric["Convert builds_per_day to numeric"]
    toNumeric --> dropNa["Drop rows missing builds_per_day"]
    dropNa --> returnDf(["Return DataFrame"])
```

## `_normalize_to_date`
```mermaid
graph TD
    start(["Start"]) --> toDatetime["Convert to UTC timestamps"]
    toDatetime --> normalize["Normalize to midnight"]
    normalize --> dropTz["Remove timezone"]
    dropTz --> returnSeries(["Return Series"])
```

## `_prepare_schedule_for_waste_analysis`
```mermaid
graph TD
    start(["Start"]) --> empty{"df is empty?"}
    empty -- Yes --> returnEmpty(["Return df"])
    empty -- No --> copy["Copy to schedule"]
    copy --> hasTs{"Contains merge_date_ts?"}
    hasTs -- Yes --> pickTs["date_series = merge_date_ts"]
    hasTs -- No --> hasDate{"Contains merge_date?"}
    hasDate -- Yes --> pickDate["date_series = merge_date"]
    hasDate -- No --> setNaT["Set schedule_date = NaT"] --> returnFallback(["Return schedule"])
    pickTs --> normalize
    pickDate --> normalize
    normalize["Normalize to schedule_date"] --> convertBuilds["Coerce scheduled_additional_builds to float"]
    convertBuilds --> cleanProject["Trim project names"]
    cleanProject --> dropInvalid["Drop rows missing project or schedule_date"]
    dropInvalid --> returnPrepared(["Return prepared schedule"])
```

## `_safe_ratio`
```mermaid
graph TD
    start(["Start"]) --> zero{"denominator == 0?"}
    zero -- Yes --> returnNan(["Return NaN"])
    zero -- No --> divide(["Return numerator / denominator"])
```

## `_baseline_detection_metrics`
```mermaid
graph TD
    start(["Start"]) --> groupMedian["Group detection_df by project, median days"]
    groupMedian --> resetIndex["Reset index"]
    resetIndex --> merge["Merge with build_counts_df"]
    merge --> fillRate["Fill builds_per_day NaN with 0"]
    fillRate --> computeBuilds["baseline_detection_builds = days * builds_per_day"]
    computeBuilds --> returnDf(["Return DataFrame"])
```

## `_build_threshold_map`
```mermaid
graph TD
    start(["Start"]) --> init["thresholds = {}"]
    init --> loop{{For each row in baseline_df}}
    loop --> readProject["project = stripped name"]
    readProject --> hasProject{"project is empty?"}
    hasProject -- Yes --> skip(["Continue"])
    hasProject -- No --> primary["baseline_builds = value"]
    primary --> primaryValid{"baseline_builds > 0?"}
    primaryValid -- Yes --> usePrimary["threshold = baseline_builds"]
    primaryValid -- No --> fallback["Test days * builds_per_day" ]
    fallback --> fallbackFound{"Finite positive candidate?"}
    fallbackFound -- Yes --> setCandidate["threshold = candidate"]
    fallbackFound -- No --> setInf["threshold = inf"]
    skip --> loop
    usePrimary --> store
    setCandidate --> store
    setInf --> store
    store["thresholds[project] = threshold"] --> loop
    loop --> done(["Return thresholds"])
```

## `_prepare_project_metrics`
```mermaid
graph TD
    start(["Start"]) --> init["project_frames = []"]
    init --> loop{{For each strategy-schedule pair}}
    loop --> summarize["summary = summarize_schedule_by_project"]
    summarize --> empty{"summary empty?"}
    empty -- Yes --> appendEmpty["Append summary"] --> loop
    empty -- No --> merge["Merge with baseline_df"]
    merge --> rename["Rename columns"]
    rename --> avg["Compute avg_builds_per_trigger"]
    avg --> fillNa["Fill baseline columns with 0"]
    fillNa --> estimateDays["estimated_detection_days = max(day delta, 0)"]
    estimateDays --> estimateBuilds["estimated_detection_builds = max(build delta, 0)"]
    estimateBuilds --> append["Append merged frame"]
    append --> loop
    loop --> concat{"project_frames empty?"}
    concat -- Yes --> returnEmpty(["Return empty DataFrame"])
    concat -- No --> returnConcat(["Concatenate frames"])
```

## `_aggregate_strategy_metrics`
```mermaid
graph TD
    start(["Start"]) --> init["records = []"]
    init --> group{{Group project_df by strategy}}
    group --> emptyGroup{"Group empty?"}
    emptyGroup -- Yes --> nextGroup(["Continue"])
    emptyGroup -- No --> compute["Compute totals, medians, means"]
    compute --> estimateSaved["Calculate total_estimated_builds_saved"]
    estimateSaved --> append["Append record"]
    append --> group
    group --> done(["Return DataFrame"])
```

## `_prepare_daily_totals`
```mermaid
graph TD
    start(["Start"]) --> init["frames = []"]
    init --> loop{{For each strategy-schedule pair}}
    loop --> empty{"schedule empty?"}
    empty -- Yes --> continueLoop(["Continue"])
    empty -- No --> pickColumn{"Has merge_date_ts?"}
    pickColumn -- Yes --> useTs["date_column = merge_date_ts"]
    pickColumn -- No --> hasDate{"Has merge_date?"}
    hasDate -- Yes --> useDate["date_column = merge_date"]
    hasDate -- No --> skipSchedule(["Skip schedule"])
    useTs --> normalize
    useDate --> normalize
    normalize["Convert to datetime; drop NaT"] --> groupSum["Group by project + date sum"]
    groupSum --> tagStrategy["Insert strategy column"]
    tagStrategy --> append["Append frame"]
    append --> loop
    loop --> framesEmpty{"frames empty?"}
    framesEmpty -- Yes --> returnEmpty(["Return empty DataFrame"])
    framesEmpty -- No --> concat["Concatenate frames"]
    concat --> sort["Sort by strategy, project, date"]
    sort --> returnDf(["Return result"])
```

## `_summarize_wasted_builds`
```mermaid
graph TD
    start(["Start"]) --> precompute["threshold map, baseline rates, detection window"]
    precompute --> init["summary_records = [] and event_records = []"]
    init --> strategyLoop{{For each strategy-schedule pair}}
    strategyLoop --> prepare["Normalize schedule"]
    prepare --> empty{"prepared empty?"}
    empty -- Yes --> appendZero["Append zero metrics"] --> strategyLoop
    empty -- No --> sort["Sort by project and date"]
    sort --> totals["Sum scheduled_additional_builds"]
    totals --> projectLoop{{For each project group}}
    projectLoop --> setup["Load threshold and baseline rate; reset counters"]
    setup --> eventLoop{{For each row ordered by date}}
    eventLoop --> advanceBaseline["Advance baseline progress by elapsed days"]
    advanceBaseline --> trimWindow["Trim baseline history beyond detection window"]
    trimWindow --> baselineDetect{"Baseline meets threshold?"}
    baselineDetect -- Yes --> markBaseline["Set detected_state = baseline"]
    baselineDetect -- No --> continueState
    markBaseline --> continueState
    continueState["Init classification, consumed, wasted"] --> stateCheck{"detected_state"}
    stateCheck -- baseline --> baselineOnly["classification = baseline_only"]
    stateCheck -- additional --> fpPost["classification = fp_post_detection"]
    stateCheck -- other --> thresholdCheck{"Threshold finite?"}
    thresholdCheck -- No --> markFP["classification = fp"]
    thresholdCheck -- Yes --> reach{"Progress + scheduled >= threshold?"}
    reach -- Yes --> markTP["Compute consumed; set classification = tp; update counters"]
    reach -- No --> markFP
    baselineOnly --> record
    fpPost --> record
    markTP --> record
    markFP --> record
    record["Append event record; update last_date"] --> eventLoop
    eventLoop --> projectDone(["Project complete"])
    projectDone --> aggregate["Compute trigger counts, ratios, project sets"]
    aggregate --> appendSummary["Append summary record"]
    appendSummary --> strategyLoop
    strategyLoop --> finalize["Create summary_df and events_df"]
    finalize --> return(["Return summary_df, events_df"])
```

## `_plot_additional_builds_boxplot`
```mermaid
graph TD
    start(["Start"]) --> empty{"project_df empty?"}
    empty -- Yes --> returnNone(["Return None"])
    empty -- No --> group["Group by strategy"]
    group --> collect{"Collected data empty?"}
    collect -- Yes --> returnNone
    collect -- No --> plot["plt.boxplot"]
    plot --> label["Set labels, title, grid"]
    label --> save["Save figure"]
    save --> close["plt.close"]
    close --> returnPath(["Return file path"])
```

## `parse_args`
```mermaid
graph TD
    start(["Start"]) --> init["Create ArgumentParser"]
    init --> addPred["Add --predictions-root"]
    addPred --> addRiskCol["Add --risk-column"]
    addRiskCol --> addLabel["Add --label-column"]
    addLabel --> addThreshold["Add --risk-threshold"]
    addThreshold --> addDetection["Add --detection-table"]
    addDetection --> addBuilds["Add --build-counts"]
    addBuilds --> addOutput["Add --output-dir"]
    addOutput --> addWindow["Add --detection-window-days"]
    addWindow --> addSilent["Add --silent flag"]
    addSilent --> parse["Parse arguments"]
    parse --> returnArgs(["Return Namespace"])
```

## `main`
```mermaid
graph TD
    start(["Start"]) --> args["args = parse_args"]
    args --> outDir["Ensure output directory"]
    outDir --> detection["Load detection table"]
    detection --> buildCounts["Load build counts"]
    buildCounts --> baseline["Compute baseline metrics"]
    baseline --> window["Clamp detection window >= 0"]
    window --> simulate["Run minimal simulation"]
    simulate --> enrich["Augment summary with detection baseline"]
    enrich --> projectMetrics["Prepare project metrics"]
    projectMetrics --> aggregateMetrics["Aggregate strategy metrics"]
    aggregateMetrics --> dailyTotals["Prepare daily totals"]
    dailyTotals --> wastedMetrics["Summarize wasted builds"]
    wastedMetrics --> write["Write CSVs and plot"]
    write --> silent{"args.silent?"}
    silent -- Yes --> endSilent(["End"])
    silent -- No --> printNode["Print summaries and paths"]
    printNode --> finish(["End"])
```
