using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;
using UnityEngine.XR;
using Valve.VR;


public class AdaptRePlay : BaseGuidance  // ガイダンスに関する計算・処理を行う。
{
    // 定数を定義して読みやすくする
    private const int STOP_THRESHOLD = 135; // 1.5秒停止と判断するカウント
    private const int DEFAULT_LEVEL_OFFSET = 5;
    private const int DEFAULT_TRIAL_OFFSET = 45;
    
    // 時間および位置管理
    private float trialTime = 0f;
    private int correspondTime = 0;  // -1: 試行間
    private int guidanceTime = 0;    // -1: ユーザーが右端到達
    
    
    // オフセット関連
    private int trialOffset = DEFAULT_TRIAL_OFFSET;
    private int levelOffset = DEFAULT_LEVEL_OFFSET;
    private int userLevel = 2;
    
    // 再生制御関連
    private int updateCount;
    private int stopCount = 0;
    private float distToFile = 0f;
    private int condition;
    
    // VR入力関連
    private SteamVR_Action_Boolean Iui = SteamVR_Actions.default_InteractUI;
    private Boolean interactUI;
    private SteamVR_Action_Boolean GrabG = SteamVR_Actions.default_GrabGrip;
    private Boolean grabGrip;
    private Vector3 RightHandPosition;
    private Quaternion RightHandRotationQ;
    
    // オブジェクト参照
    private GameObject stopUser, stopGuidance, replayGuidance;
    private GameObject wristR;

    // 難易度マップを格納する配列（各フレームの精度情報）
    private float[] accuracyMap;
    private float[] preAccuracyMap;  // ファイルに書き込むため、一つ前の動作終了後の精度マップを保存。
    private const float DEFAULT_ACCURACY = 1.0f;
    private const float MAX_ACCURACY = 1.0f;
    private const float MIN_GUIDANCE_SPEED = 0.1f;

    // プロパティ定義
    public int CorrespondTime => correspondTime;
    public int GuidanceTime => guidanceTime;
    public int TrialOffset => trialOffset;
    public int LevelOffset => levelOffset;
    public int UserLevel => userLevel;
    public float DistToFile => distToFile;
    public Boolean InteractUI => interactUI;
    public float[] AccuracyMap => accuracyMap;
    public float[] PreAccuracyMap => preAccuracyMap;


    

    // 定数を追加
    private const int USER_LEAD_FRAMES = 5; // ユーザーがこのフレーム数進んだ後にガイダンスが動き始める
    private const float GUIDANCE_FOLLOW_RATE = 0.8f; // ガイダンスの基本追従率（1.0で完全に追いつく）

    // 累積値を保持する変数を追加
    private float accumulatedDifficulty = 0f;

    public AdaptRePlay(GameObject guidance, GameObject user, int fileRowCount, Vector3[] positions, Quaternion[] quaternions, Material[] materialArray, 
    int experiment4_condition, GameObject stopUser, GameObject stopGuidance, GameObject replayGuidance, GameObject wristR)
        : base(guidance, user, fileRowCount, positions, quaternions, materialArray)
    {
        condition = experiment4_condition;
        this.stopUser = stopUser;
        this.stopGuidance = stopGuidance;
        this.replayGuidance = replayGuidance;
        this.wristR = wristR;
        
        // 難易度マップの初期化
        accuracyMap = new float[fileRowCount];
        preAccuracyMap = new float[fileRowCount+1];
        for (int i = 0; i < fileRowCount; i++)
        {
            accuracyMap[i] = DEFAULT_ACCURACY;
            preAccuracyMap[i] = DEFAULT_ACCURACY;
        }
        preAccuracyMap[fileRowCount] = 0;  // 0は、精度に関係ない数値であることを表す。書き込みファイルの0を消すことで、正しい精度マップを得られる。
    }
    

    
    // 現フレームのユーザーの精度を評価
    public override float Evaluation()
    {
        int nearest = 0;
        float minDist = 500f;
        int searchRange = 45;

        // ファイル範囲を超えないよう調整
        if (correspondTime + searchRange > fileRowCount - 1)
        {
            searchRange = fileRowCount - 1 - correspondTime;
        }

        // 最も近い位置を検索
        for (int jump = 0; jump <= searchRange; jump++)
        {
            float dist = Vector3.Distance(RightHandPosition, modelPositions[correspondTime + jump]);
            if (dist < minDist)
            {
                minDist = dist;
                nearest = jump;
            }
        }
        
        // ユーザー位置を更新
        correspondTime += nearest;
        
        // 難易度マップを更新（精度が低いほど高い難易度値）
        //float difficultyValue = CalculateDifficulty(minDist);
        //UpdateDifficultyMap(nearest, correspondTime, difficultyValue);
        
        return 0;
    }
    
    // 精度を計算する関数（距離に基づく）
    // 距離が大きいほど小さい値を返す
    // distance = 0のとき1.0、distance = 0.1以上のとき0.1を返す
    private float CalculateDifficulty(float distance)
    {
        float normalizedDist = Math.Min(distance / 0.3f, 1.0f);
        // 1.0から0.1の範囲で反転させる
        return 1.0f - (normalizedDist * 0.9f);
    }
    
    // 難易度マップを更新
    private void UpdateDifficultyMap(int skip, int position, float newDifficulty)
    {
        if (position < accuracyMap.Length)
        {
            int prePosition = position - skip;
            // 既存の難易度と新しい難易度の平均を取る
            accuracyMap[position] = (accuracyMap[position] + newDifficulty) / 2f;
            
            float leftDiff = accuracyMap[prePosition];
            float rightDiff = accuracyMap[position];
            
            // skipに依存するminDifficultyの計算
            float skipFactor = 1.0f - 0.0111f * skip;  // skipがその最大値である45をとると、0.5になる。
            float minDifficulty = Math.Max(0.1f, Math.Min(leftDiff, rightDiff) * skipFactor);

            for(int i = 1; i < skip; i++)
            {
                float t = (float)i / skip; // 0から1の進行度

                // ベジエ曲線による難易度の計算
                float bezierValue = 
                    (1 - t) * (1 - t) * leftDiff +
                    2 * (1 - t) * t * minDifficulty +
                    t * t * rightDiff;

                accuracyMap[prePosition + i] = bezierValue;
            }
        }
    }
    
    // ユーザー停止時の処理を切り出し
    private void HandleUserStop(int nearest)
    {
        // 停止時には何もしない（ガイダンスはユーザーを追いかけ続ける）
        stopCount = 0;
    }
    
    public override void Moving(int updateCount)
    {
        // ガイダンスがユーザーを追い越さないようにする
        if (guidanceTime >= correspondTime - 5 || guidanceTime == -1)
        {
            return;
        }

        // 現在位置の難易度値を取得して累積
        float currentDifficulty = accuracyMap[guidanceTime];
        accumulatedDifficulty += currentDifficulty;

        // Debug.Log($"Accumulated: {accumulatedDifficulty}, Difficulty: {currentDifficulty}"); // デバッグ用

        // 累積値が1を超えたら進む
        while (accumulatedDifficulty >= 1.0f)
        {
            guidanceTime++;
            accumulatedDifficulty -= 1.0f;
        }
        
        // ガイダンス終了判定
        if (guidanceTime >= fileRowCount - 10)
        {
            HandleTrialCompletion();
        }
    }

    // 難易度に基づいて速度係数を取得（よりシンプルな実装）
    private float GetSpeedFactor(int position)
    {
        if (position < 0 || position >= accuracyMap.Length)
        {
            return 1.0f;
        }
        
        // 難易度が高いほど速度が遅くなる（1.0～0.2）
        float difficulty = accuracyMap[position];
        return Math.Max(MIN_GUIDANCE_SPEED, 1.0f / difficulty);
    }

    // 試行完了処理を切り出し
    private void HandleTrialCompletion()
    {
        Debug.Log("HandleTrialCompletion\n" + "guidanceTime: " + guidanceTime);
        for (int i = 0; i < accuracyMap.Length; i++)
        {
            preAccuracyMap[i] = accuracyMap[i];
        }
        guidanceTime = -1;
    }
    
    public override void GuidanceUpdate()
    {
        // VR入力の更新
        UpdateVRInput();

        // マウスの右クリックを検知してデバッグログを出力
        if (Input.GetMouseButtonDown(1))
        {

            // 難易度マップをデバッグログに出力
            Debug.Log("難易度マップの内容:");
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < accuracyMap.Length; i++)
            {
                sb.AppendFormat("フレーム {0}: {1:F2}, ", i, accuracyMap[i]);
                if ((i + 1) % 5 == 0)
                {
                    Debug.Log(sb.ToString());
                    sb.Clear();
                }
            }
            if (sb.Length > 0)
            {
                Debug.Log(sb.ToString());
            }
        }

        if (interactUI)
        {
            trialTime += Time.deltaTime;

            if (correspondTime == -1 && guidanceTime == -1)
            {
                InitializeNewTrial();
            }
            else if(guidanceTime != -1)  // ガイダンスが最後まで行くと、HandleTrialComplition()が呼び出され、guidanceTime=-1となる。
            {
                // 通常の更新処理
                stopUser.transform.position = Vector3.zero;
                stopGuidance.transform.position = Vector3.zero;
                
                updateCount++;
                
                // ユーザーの精度評価と位置更新
                Evaluation();
                
                // ユーザーが十分進んでいれば、ガイダンスを動かす
                if (correspondTime >= USER_LEAD_FRAMES)
                {
                    Moving(1); // 毎フレーム呼び出す
                }
            }
        }
        // guidanceTimeがきちんとHandleTrialComplition()によって－1になっている状況で、トリガーを外した場合、正常終了を表すためにcorrespondTime=-1にする。
        else if (guidanceTime == -1)  
        {
            correspondTime = -1;  // 試行正常終了
        }
        else
        {
            HandleTrialInterruption();
        }
    }

    // VR入力更新処理
    private void UpdateVRInput()
    {
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);

        InputDevice rightHandDevice = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (rightHandDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 newRightHandPosition))
        {
            RightHandPosition = newRightHandPosition;
        }
        if (rightHandDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion newRightHandRotationQ))
        {
            RightHandRotationQ = newRightHandRotationQ;
        }
        RightHandPosition = SteamVR_Behaviour_Skeleton.ControllerOffsetPos(RightHandPosition, RightHandRotationQ);
        RightHandRotationQ = SteamVR_Behaviour_Skeleton.ControllerOffsetRot(RightHandPosition, RightHandRotationQ);
    }

    // 新規試行の初期化を修正
    private void InitializeNewTrial()
    {
        // ガイダンスは最初は動かない、ユーザーが先行
        correspondTime = 0;
        guidanceTime = 0; // ガイダンスは0から開始
        trialTime = 0f;
        updateCount = 0;
        
        // // 難易度マップをリセット
        // for (int i = 0; i < accuracyMap.Length; i++)
        // {
        //     accuracyMap[i] = DEFAULT_ACCURACY;
        // }
        
        accumulatedDifficulty = 0f;
    }

    // 試行中断処理
    private void HandleTrialInterruption()
    {
        Debug.Log("トリガーが外れました");

        for (int i = 0; i < accuracyMap.Length; i++)
        {
            preAccuracyMap[i] = 0;  // 0は、精度に関係ない数値であることを表す。書き込みファイルの0を消すことで、正しい精度マップを得られる。
        }
        guidanceTime = -1;
        correspondTime = -1;
    }
}

