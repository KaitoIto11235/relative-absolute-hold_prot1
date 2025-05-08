using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;

public class AutoCondition : MonoBehaviour
{
    [SerializeField] AudioSource audioSource;
    [SerializeField] private GameObject guidance, user;
    [SerializeField] string readFileName = "default";
    [SerializeField] string writeFileName = "default";
    [SerializeField] GameObject startPoint, startPoint2, endPoint, endPoint2;
    [SerializeField] int readFileRowCount = 1000;
    FileOperation autoFile;
    AutoPlay autoGuidance;
    [SerializeField] bool Recording = false;
    [SerializeField] Material[] materialArray = new Material[3];
    int commaPlaySpeed = 10; // 10が等速再生
    //[SerializeField, Range(1, 20)] int commaPlaySpeed = 10;

    [SerializeField] GameObject wristR;
    [SerializeField] Animator p_Animator;
    [SerializeField] bool is3PP;

    [Tooltip("If not set, relative to parent")]
    public Transform origin;

    [Header("Material Settings")] // インスペクタで見やすくするためのヘッダー
    [SerializeField] private Material materialOdd; // 奇数試行用のマテリアル
    [SerializeField] private Material materialEven; // 偶数試行用のマテリアル
    private string betaJointsObjectName = "Beta_Joints"; // 子オブジェクトの名前
    private string betaSurfaceObjectName = "Beta_Surface"; // 子オブジェクトの名前

    private Renderer betaJointsRenderer; // Beta_Jointsのレンダラーコンポーネント
    private Renderer betaSurfaceRenderer; // Beta_Surfaceのレンダラーコンポーネント
    private int currentTrialParity = -1; // 現在の試行の偶奇性 (0:偶数, 1:奇数, -1:未設定)

    void Start()
    {
        // "Beta_Joints" オブジェクトを探してレンダラーを取得
        Transform betaJointsTransform = transform.Find(betaJointsObjectName);
        Transform betaSurfaceTransform = transform.Find(betaSurfaceObjectName);
        if (betaJointsTransform != null && betaSurfaceTransform != null)
        {
            betaJointsRenderer = betaJointsTransform.GetComponent<Renderer>();
            betaSurfaceRenderer = betaSurfaceTransform.GetComponent<Renderer>();
            if (betaJointsRenderer == null || betaSurfaceRenderer == null)
            {
                Debug.LogError($"'{betaJointsObjectName}'または'{betaSurfaceObjectName}' オブジェクトに Renderer コンポーネントが見つかりません。", this);
            }
        }
        else
        {
            Debug.LogError($"子オブジェクト '{betaJointsObjectName}'または'{betaSurfaceObjectName}' が見つかりません。", this);
        }

        // --- FileOperationの初期化方法を変更 ---
        // 1. 新しいコンストラクタでインスタンス作成
        try
        {
            autoFile = new FileOperation(user, startPoint, startPoint2, endPoint, endPoint2);
        }
        catch (Exception ex)
        {
             Debug.LogError($"Failed to instantiate FileOperation: {ex.Message} {ex.StackTrace}", this);
             enabled = false; // エラー時はコンポーネントを無効化
             return;
        }

        // 2. モデルデータ読み込み (ファイル名が指定されている場合のみ)
        bool modelLoaded = false;
        if (!string.IsNullOrEmpty(readFileName))
        {
            if (!autoFile.LoadModelData(readFileName, readFileRowCount))
            {
                 Debug.LogError($"Failed to load model data: {readFileName}. Disabling component.", this);
                 enabled = false;
                 return;
            }
            else
            {
                 // AutoPlayの初期化に必要なreadFileRowCountを、実際に読み込んだ行数に更新
                 // (LoadModelData内で不一致時に更新される可能性があるため)
                 readFileRowCount = autoFile.FileRowCount;
                 modelLoaded = true;
            }
        }
        else
        {
             Debug.LogWarning("readFileName is not set. Skipping model loading.", this);
             // モデルがない場合、AutoPlayの初期化は行わない
        }

        // 3. 書き込みファイル名設定 (Recordingがtrueの場合のみ)
        if (Recording)
        {
            if (!string.IsNullOrEmpty(writeFileName))
            {
                autoFile.SetWriteFileNameBase(writeFileName);
            }
            else
            {
                 Debug.LogError("Recording is enabled but writeFileName is not set. Recording might fail.", this);
                 // 必要なら enabled = false; return; などで停止
            }
            // WriteOpenData は RecordingUpdate 内で必要に応じて呼ばれるため、ここでは呼ばない
        }

        // --- ここまで FileOperation 初期化変更 ---

        // --- AutoPlayの初期化 --- (モデルが正常にロードされた場合のみ)
        // 古い初期化ブロックを削除 または コメントアウト
        /*
        if (autoFile != null && !string.IsNullOrEmpty(readFileName))
        {
            autoFile.ReadOpenData(); // -> LoadModelData に置き換え済み
            // ReadOpenDataが成功したか、modelPositionsがnullでないか確認
            if (autoFile.modelPositions != null && autoFile.modelQuaternions != null)
            { ... }
        }
        */
        if (modelLoaded && autoFile != null)
        {
            try
            {
                // AutoPlayのコンストラクタ引数から materialArray を削除 (AutoPlay側で修正済みのはず)
                // FileRowCountは実際にロードされた行数を使う
                autoGuidance = new AutoPlay(guidance, user, autoFile.FileRowCount, autoFile.modelPositions, autoFile.modelQuaternions,
                                       commaPlaySpeed, /* materialArray, */ wristR, p_Animator);
                Debug.Log("AutoPlay initialized successfully.");
            }
            catch (Exception ex)
            {
                 Debug.LogError($"Failed to initialize AutoPlay: {ex.Message} {ex.StackTrace}", this);
                 enabled = false;
                 return;
            }
        }
        else if (autoFile == null)
        {
            Debug.LogError("FileOperation instance (autoFile) is null after initialization attempt.", this);
            enabled = false;
            return;
        }
        // readFileName が空で modelLoaded が false の場合は AutoPlay を初期化しない（上のログで警告済み）


        // 初期マテリアルを設定
        UpdateBetaJointsMaterial();
    }

    void FixedUpdate()
    {
        if (autoFile.fileSettingWrong)
        {
            Debug.LogError("File setting check failed after initialization. Disabling component.", this);
            enabled = false;
            OnDestroy();
            UnityEditor.EditorApplication.isPlaying = false;
            return;
        }
        if(autoGuidance != null)
        {
            autoGuidance.GuidanceUpdate();
        }

        // 効果音 (GuidanceTime を CurrentGuidanceTime に変更)
        if(autoGuidance != null && (autoGuidance.CurrentGuidanceTime+1) % 90 == 0 && autoGuidance.CurrentGuidanceTime < 721 && autoGuidance.CurrentGuidanceTime > -1)
        {
            audioSource.Play();
        }

        // RecordingUpdate はユーザーの位置更新と記録状態の管理を行うため、そのまま呼び出す
        if (Recording && autoFile != null)
        { 
             autoFile.RecordingUpdate();
 
            // TrialCount の偶奇性が変化したらマテリアルを更新 (変更なし)
            UpdateBetaJointsMaterial();
        }
    }

    // Beta_Joints のマテリアルを更新するメソッド
    void UpdateBetaJointsMaterial()
    {
        if (betaJointsRenderer == null || betaSurfaceRenderer == null || autoFile == null) // レンダラーまたはautoFileがなければ何もしない
        {
            return;
        }

        int newParity = autoFile.CurrentTrialCount % 2; // FileOperation側のプロパティ名変更に合わせる

        // 偶奇性が変化した場合のみマテリアルを更新
        if (newParity != currentTrialParity)
        {
            Material targetMaterial = (newParity == 1) ? materialOdd : materialEven;

            if (targetMaterial != null)
            {
                betaJointsRenderer.material = targetMaterial;
                betaSurfaceRenderer.material = targetMaterial;
                currentTrialParity = newParity; // 現在の偶奇性を更新
                // Debug.Log($"Trial {autoFile.CurrentTrialCount}: Set material to {(newParity == 1 ? "Odd" : "Even")}");
            }
            else
            {
                 Debug.LogWarning($"Trial {autoFile.CurrentTrialCount}: {(newParity == 1 ? "materialOdd" : "materialEven")} が設定されていません。", this);
            }
        }
    }

    void OnAnimatorIK()
    {
        if (autoGuidance == null || p_Animator == null) return; // nullチェックを追加

        Vector3 TargetPos = Vector3.zero; // 初期化
        Quaternion TargetRot = Quaternion.identity; // 初期化

        // guidanceTime の取得箇所で CurrentGuidanceTime を使う
        int guidanceTime = (autoGuidance != null) ? autoGuidance.CurrentGuidanceTime : 0; // Nullチェック & プロパティ名変更
        int modelIndex = -1; // 初期化
        // FileOperation側のプロパティ名変更に合わせて FileRowCount を使う
        if (autoFile != null && autoFile.modelPositions != null && autoFile.FileRowCount > 0) {
             modelIndex = Math.Min(Math.Max(0, guidanceTime), autoFile.FileRowCount - 1); // 配列長ではなくFileRowCountで制限
        } else {
            // Debug.LogWarning("modelPositions is null or FileRowCount is zero, cannot calculate modelIndex.");
             return; // 配列がないか行数が0ならIK設定をスキップ
        }


        // ModelPositions と ModelQuaternions が null でなく、インデックスが有効な範囲にあるか再確認
        if (modelIndex != -1 && autoFile.modelQuaternions != null && modelIndex < autoFile.modelQuaternions.Length)
        {
            if(origin == null)
            {
                float three = 0;
                if(is3PP == true)
                {
                    three = 3;
                }
                Vector3 offset = new Vector3(0, 0, three);
                TargetPos = autoFile.modelPositions[modelIndex] + offset; // autoGuidance経由ではなくautoFileから直接アクセス
                TargetRot = autoFile.modelQuaternions[modelIndex]; // autoGuidance経由ではなくautoFileから直接アクセス
            }
            else
            {
                TargetPos = origin.TransformPoint(autoFile.modelPositions[modelIndex]);
                TargetRot = origin.rotation * autoFile.modelQuaternions[modelIndex];
            }
        } else {
             // データがない場合のフォールバック処理
             // Debug.LogWarning($"IK target data is not available for index {modelIndex}.");
             return; // データがない場合はIK設定をスキップ
        }

        // IK設定 (既存のコード)
        p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1.0f);
        p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 1.0f);
        p_Animator.SetIKPosition(AvatarIKGoal.RightHand, TargetPos);
        p_Animator.SetIKRotation(AvatarIKGoal.RightHand, TargetRot);
    }

    void OnDestroy()
    {
        if (autoFile != null && Recording)
        {
           autoFile.CloseFile();
        }
    }
}


