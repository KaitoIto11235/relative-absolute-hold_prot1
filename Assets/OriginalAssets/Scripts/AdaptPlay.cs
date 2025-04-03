using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;
using UnityEngine.XR;
using Valve.VR;

public interface IGuidance
{
    float Evaluation();
    void Moving(int updateCount);
    void GuidanceUpdate();
}
public abstract class BaseGuidance : IGuidance
{
    protected GameObject user, guidance;
    protected int fileRowCount;
    protected Vector3[] modelPositions;
    public Vector3[] ModelPositions
    {
        get {return modelPositions;}
    }
    protected Quaternion[] modelQuaternions;
    public Quaternion[] ModelQuaternions
    {
        get {return modelQuaternions;}
    }
    protected Material[] materialArray;

    public BaseGuidance(GameObject guidance, GameObject user, int fileRowCount, Vector3[] positions, Quaternion[] quaternions, Material[] materialArray)
    {
        this.guidance = guidance;
        this.user = user;
        this.fileRowCount = fileRowCount;
        this.modelPositions = new Vector3[fileRowCount];
        this.modelPositions = positions;
        this.modelQuaternions = new Quaternion[fileRowCount];
        this.modelQuaternions = quaternions;
        this.materialArray = materialArray;
    }

    public abstract float Evaluation();
    public abstract void Moving(int updateCount);
    public abstract void GuidanceUpdate();
}

public class AdaptPlay : BaseGuidance  // ガイダンスに関する計算・処理を行う。
{
    float trialTime = 0f;        // 1試行の時間
    private int availableNum = 5, notAvailableNum = 0;
    private int correspondTime = 0;  // Userの現在地に対応するModelの時間。 値が-1のとき、試行と試行の間であることを意味する
    public int CorrespondTime
    {
        get {return correspondTime;}
    }
    private int offsetCorrespondTime = 0;  // Userの現在地に、trialScoreを加えたもの。
    private int guidanceTime = 0;   // ガイダンスの現在の時間。値が-1のとき、ユーザーが右端まで到達したことを意味する
    public int GuidanceTime
    {
        get {return guidanceTime;}
    }
    private float frame_5_score = 0f;       // 5フレームでのスコア
    private int trialOffset = 45; // 1試行終了時にガイダンスが何フレーム離れているか
    public int TrialOffset
    {
        get{return trialOffset;}
    }
    private int trialOffsetAverage = 0;  // 1練習の中における、trialOffsetの平均
    private int trialOffsetStock = 0;  // 1練習の中で、trialOffset>=5のときに増加する
    private int tiralOffsetCount = 0;  // 1練習の中で、trialOffset>=5のときの回数
    private int levelOffset = 5; // 現レベルにおいて、ガイダンスが何フレーム離れているか。trialOffsetの合計。
    public int LevelOffset
    {
        get{return levelOffset;}
    }
    private int  userLevel = 2;  // ユーザーのレベル 
    public int UserLevel
    {
        get {return userLevel;}
    }
    private int updateCount;
    private int stopCount = 0;   // 同じ対応点で停止している時間
    private bool initialOperation = true;

    private int playBack = 0;  // 再検索時、tフレームからt－1フレームのユーザー位置まで何コマ戻るか。

    //private int fileRowCount;
    //private GameObject user, guidance;
    //private Vector3[] modelPositions;

    private float distToFile = 0f;  // ファイルに1フレームの誤差を記録するための変数（correspondTimeが更新されたときに限る）
    public float DistToFile
    {
        get {return distToFile;}
    }

    private int condition;  // 実験条件

    // InteractUIボタンが押されているかを判定するためのIuiという関数にSteamVR_Actions.defalt_InteractionUIを固定
    private SteamVR_Action_Boolean Iui = SteamVR_Actions.default_InteractUI;
    // 結果の格納用Boolean型変数interacrtui
    private Boolean interactUI;
    public Boolean InteractUI
    {
        get {return interactUI;}
    }

    // GrabGripボタン（初期設定は側面ボタン）が押されているかを判定するためのGrabという関数にSteamVR_Actions.defalt_GrtabGripを固定
    private SteamVR_Action_Boolean GrabG = SteamVR_Actions.default_GrabGrip;
    // 結果の格納用Boolean型関数grabgrip;
    private Boolean grabGrip;
    private Vector3 RightHandPosition;
    private Quaternion RightHandRotationQ;
    // User停止時に手の上に表示されるオブジェクト
    private GameObject stopUser, stopGuidance, replayGuidance;
    private GameObject wristR;
    private int researchCount = 0;

    public AdaptPlay(GameObject guidance, GameObject user, int fileRowCount, Vector3[] positions, Quaternion[] quaternions, Material[] materialArray, 
    int experiment4_condition, GameObject stopUser, GameObject stopGuidance, GameObject replayGuidance, GameObject wristR)
        : base(guidance, user, fileRowCount, positions, quaternions, materialArray)
    {
        condition = experiment4_condition;
        this.stopUser = stopUser;
        this.stopGuidance = stopGuidance;
        this.replayGuidance = replayGuidance;
        this.wristR = wristR;
    }
    
    // float JumpPenalty(int jump, int stopCount) // 前フレームの対応点から2つ以上離れた点に対して、距離に対してペナルティを与える関数
    // {
    //     if(jump == 0 || jump == 1 || stopCount > 25)
    //     {
    //         return 0f;
    //     }
    //     else
    //     {
    //         return (25 - stopCount) * 0.01f * jump;
    //     }
    // }

    float FrameScore(float minDist, int stopCount, int level, int condition) // 今回の呼び出しにおいて、最も近い点との距離minDiffをスコア化する関数
    {
        // if(stopCount > 25)
        // {
        //     return 0.2f;
        // }
        if(condition == 1 || condition == 4)
        {
            return 1.2f;
        }
        else if(condition == 2 || condition == 5)
        {
            return 1.5f;
        }
        else if(condition == 3 || condition == 6)
        {
            return 2.0f;
        }
        else // error
        {
            Debug.Log("Condition_Error");
            return 0f;
        }
    }
    
    // 現フレームのユーザーの精度を評価
    public override float Evaluation()
    {
        int nearest = 0;        // 今回の呼び出しで対応点のインデックスがどれだけ進むか


        // 現ユーザーのポジションを評価
        if(offsetCorrespondTime <= guidanceTime)
        {
            float dist = 0f;
            float minDist = 500f;
            int searchRange = 0;

            if(stopCount < 135)  // 1.5秒停止しなかったら
            {
                //searchRange = guidanceTime - offsetCorrespondTime;
                searchRange = Math.Min(5, guidanceTime - offsetCorrespondTime);
            }
            else  // 学習者が軌道を大きく外れ、ガイダンス位置についてしまい、更新されなくなったとき用。
            {
                searchRange = guidanceTime - correspondTime;
                researchCount++;
            }

            if(correspondTime + searchRange > fileRowCount - 1)
            {
                searchRange = fileRowCount - 1 - correspondTime;
            }

            //offsetCorrespondTimeからguidanceTimeまでの幅maxIndex-offsetCorrespondTimeだけ、現地点（correspondTime）から探索。
            for(int jump = 0; jump <= searchRange; jump++)
            {
                //dist = Vector3.Distance(RightHandPosition, modelPositions[correspondTime + jump]) + JumpPenalty(jump, stopCount);
                dist = Vector3.Distance(RightHandPosition, modelPositions[correspondTime + jump]);
                if(dist < minDist)
                {
                    minDist = dist;     // 現フレームにおけるユーザー位置のズレの最小値を更新
                    nearest = jump;     // 最小値をとるモデル位置と現ユーザー位置のindex差を更新
                }
            }
            correspondTime += nearest;
            offsetCorrespondTime += nearest;
            
            if(stopCount >= 135)// 1.5秒停止したら
            {
                initialOperation = true;
                playBack = nearest;
                
                Debug.Log("playBack:" + playBack);
                guidanceTime = correspondTime;
                //levelOffset = ((levelOffset - 5) / 2) + 5;
                //trialOffset = Math.Max(5, trialOffset / 2);
                availableNum = levelOffset + trialOffset;

                correspondTime -= nearest;  // このフレームの進みを無かったことにする。
                offsetCorrespondTime = correspondTime + levelOffset;

                if(updateCount == 5)
                {
                    updateCount = 4;  // このフレームは無かったことに、、。こうしないと、updateCount==5のとき、すぐにavailableNumが書き換えられてしまい、initialOperationによる再生が数フレームで終わってしまう。
                }
                stopCount = 0;
                return 0f;
            }
            else if(offsetCorrespondTime == guidanceTime) // このとき、次の5フレームでガイダンスが5フレーム以上進んでくれないと、探索範囲が2点のみになり、抜け出せなくなってしまう。
            {
                float score = FrameScore(minDist, stopCount, userLevel, condition);       // スコアを返す。
                stopCount = 0;
                distToFile = minDist;    // ファイルに今フレームの誤差を記録するためのもの
                if(score > 1f)
                {
                    return score;
                }
                else
                {
                    return 1f;
                }
            }
            else if(nearest == 0)  // つまり、進んでいないとき。
            {
                stopCount++;
                distToFile = 0f;
                return 0f; 
            }
            else  // ユーザーが止まっておらず、ガイダンスに追いついていなければ、
            {
                float score = FrameScore(minDist, stopCount, userLevel, condition);       // スコアを返す。
                stopCount = 0;
                distToFile = minDist;    // ファイルに今フレームの誤差を記録するためのもの
                return score;            // スコアを返す。
            }
        }
        else
        {
            return 0f;
        }
    }
    
    
    public override void Moving(int updateCount)
    {
        if(initialOperation == true)
        {
            if(playBack != 0)
            {
                guidanceTime--;
                playBack--;
                replayGuidance.transform.position = modelPositions[Math.Min(guidanceTime, fileRowCount-1)];
            }
            else
            {
                guidanceTime++;
                availableNum--;
                stopGuidance.transform.position = modelPositions[Math.Min(guidanceTime, fileRowCount-1)];
                if(availableNum <= 0)
                {
                    initialOperation = false;
                    replayGuidance.transform.position = new Vector3(0, 0, 0);
                }
            }
            
        }
        else
        {
            guidanceTime += (int)(availableNum * updateCount / 5) - notAvailableNum;  // 今回の呼び出しで表示されるガイダンスのインデックス
            notAvailableNum = availableNum * updateCount / 5;
        }

        // if(guidanceTime != -1)
        // {
        //     guidance.transform.position += modelPositions[Math.Min(fileRowCount - 1, guidanceTime)] - wristR.transform.position;
        //     guidance.transform.rotation *= Quaternion.Inverse(wristR.transform.rotation) * modelQuaternions[Math.Min(guidanceTime, fileRowCount - 1)];
        // }
        // else
        // {
        //     guidance.transform.position += modelPositions[0] - wristR.transform.position;
        //     guidance.transform.rotation *= Quaternion.Inverse(wristR.transform.rotation) * modelQuaternions[0];
        // }
        
        if(correspondTime >= fileRowCount - 10 && guidanceTime != -1)  // どれだけ先行させられたか。ガイダンスが終わった時点で呼び出され、それ以降呼び出されない。
        {
            int still_addition = Math.Max(0, (int)frame_5_score) + (availableNum - notAvailableNum); // correspondTime=fileRowCount-1のときに、gauidanceTimeに追加されなかった分
            //int still_addition = availableNum - notAvailableNum; // correspondTime=fileRowCount-1のときに、gauidanceTimeに追加されなかった分
            //trialOffset = Math.Max(5, guidanceTime +  still_addition - offsetCorrespondTime);
            trialOffsetAverage = trialOffsetStock / tiralOffsetCount;
            guidanceTime = -1;                              // それ以降呼び出されないための処理。
        }
    }
    public override void GuidanceUpdate()
    {
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);

        // 右コントローラの姿勢を取得
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

        if(Input.GetMouseButton(1)) // デバッグ用
        {
            Debug.Log("correspondTime: " + correspondTime);
            Debug.Log("offsetCorrespondTime: " + offsetCorrespondTime);
            Debug.Log("guidanceTime: " + guidanceTime);
        }

        if (interactUI)
        {
            trialTime += Time.deltaTime;

            if(correspondTime == -1 && guidanceTime == -1)  // すべての試行の初期動作。
            {
                trialOffset = -1;  // 後でデータを見たときに、初期動作であることを分かるようにするために、あり得ない数字。
                levelOffset += Math.Max(0, trialOffsetAverage-45);
                //Debug.Log("levelOffset" + levelOffset);
                if(researchCount > 3)
                {
                    levelOffset = 5 + (levelOffset - 5) / 2;
                }
                availableNum = levelOffset + 45;          // 初期動作でガイダンスがどれだけ進むか
                correspondTime = 0;
                offsetCorrespondTime = levelOffset;    // offsetCorrespondTimeはcorrespondTimeから常にこれ以上は先行する。 
                guidanceTime = 0;
                trialTime = 0f;
                notAvailableNum = 0;
                frame_5_score = 0f;
                trialOffsetStock = 0;
                tiralOffsetCount = 0;
                initialOperation = true;
                researchCount = 0;
                updateCount = 0;
            }
            else if(initialOperation == true)
            {
                stopUser.transform.position = RightHandPosition;
                if(guidanceTime == -1)
                {
                    initialOperation = false;
                }

            }
            else if(initialOperation == false)
            {
                if(guidanceTime != -1)
                {
                    trialOffset = guidanceTime - offsetCorrespondTime;
                    if(trialOffset > 0)
                    {
                        trialOffsetStock += trialOffset;
                        tiralOffsetCount++;
                    }
                }
                stopUser.transform.position = new Vector3(0, 0, 0);
                stopGuidance.transform.position = new Vector3(0, 0, 0);

                updateCount++;
                frame_5_score += Evaluation();  // ユーザーが止まっていない かつ correspondTime < guidanceTime ⇒ 現フレームのスコアが返される。
            }
            
            if(guidanceTime != -1)
            {
                Moving(updateCount);
            }

            if(updateCount == 5 && initialOperation != true)
            {
                availableNum = Math.Max(0, (int)frame_5_score);
                notAvailableNum = 0;
                frame_5_score = 0f;
                updateCount = 0;
            }
        }
        else if(guidanceTime == -1)
        {
            correspondTime = -1;  // 1試行が正常終了したことを意味する
        }
        else    // 正常終了前に、トリガーが外れてしまった場合。
        {
            Debug.Log("トリガーが外れました");
            //guidance.transform.position = modelPositions[0];
            if(levelOffset != 0)  // 初練習のとき以外に初期化
            {
                trialOffsetAverage = 0;
            }
        
            guidanceTime = -1;
            correspondTime = -1;
        }
    }
}

