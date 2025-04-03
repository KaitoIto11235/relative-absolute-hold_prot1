using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;
using UnityEngine.XR;
using Valve.VR;


public class AdaptFeedback : BaseGuidance  // ガイダンスに関する計算・処理を行う。
{
    float trialTime = 0f;        // 1試行の時間
    private int availableNum = 5, notAvailableNum = 0;
    private int correspondTime = -1;  // Userの現在地に対応するModelの時間。 値が-1のとき、試行と試行の間であることを意味する
    public int CorrespondTime
    {
        get {return correspondTime;}
    }
    private int guidanceTime = -1;   // ガイダンスの現在の時間。値が-1のとき、ユーザーが右端まで到達したことを意味する
    public int GuidanceTime
    {
        get {return guidanceTime;}
    }
    private int trialOffset = 45; // 1試行終了時にガイダンスが何フレーム離れているか
    public int TrialOffset
    {
        get{return trialOffset;}
    }
    
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
    private bool replay = true;

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

    float maxError = 0;
    int replayPointTime = 0;
    int preCorrespondTime = 0;
    int stopCount = 0;
    float ErrorScore = 0;
    float ikiti = 10;
    bool releaseTrigerOnce = false;

    private int autoMode = 0;
    public int AutoMode
    {
        get {return autoMode;}
    }
    private bool trialFinish = true;
    public bool TrialFinish
    {
        get {return trialFinish;}
    }
    

    public AdaptFeedback(GameObject guidance, GameObject user, int fileRowCount, Vector3[] positions, Quaternion[] quaternions, Material[] materialArray, 
    int experiment4_condition, GameObject stopUser, GameObject stopGuidance, GameObject replayGuidance, GameObject wristR)
        : base(guidance, user, fileRowCount, positions, quaternions, materialArray)
    {
        condition = experiment4_condition;
        this.stopUser = stopUser;
        this.stopGuidance = stopGuidance;
        this.replayGuidance = replayGuidance;
        this.wristR = wristR;
    }
    
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
        guidanceTime = 0;  // ちゃんと進み始めたら、見本を見えなくする。


        int nearest = 0;        // 今回の呼び出しで対応点のインデックスがどれだけ進むか

        float dist = 0f;
        float penalty = 0f;
        float minDist = Vector3.Distance(RightHandPosition, modelPositions[correspondTime]);

        int searchRange = 45;
        if(correspondTime + searchRange > fileRowCount - 1)
        {
            searchRange = fileRowCount - 1 - correspondTime;
        }

        //offsetCorrespondTimeからguidanceTimeまでの幅maxIndex-offsetCorrespondTimeだけ、現地点（correspondTime）から探索。
        for(int jump = 1; jump <= searchRange; jump++)
        {
            dist = Vector3.Distance(RightHandPosition, modelPositions[correspondTime + jump]);
            if(dist <= minDist)
            {
                minDist = dist;     // 現フレームにおけるユーザー位置のズレの最小値を更新
                nearest = jump;     // 最小値をとるモデル位置と現ユーザー位置のindex差を更新
                penalty = Vector3.Distance(modelPositions[correspondTime], modelPositions[correspondTime + jump]);
            }
        }
        correspondTime += nearest;

        if(nearest == 0)  // ユーザーが止まっている場合、stopCountを1増やして、0を返す。
        {
            stopCount++;
            return 0f;
        }
        stopCount = 0;
        return minDist + penalty;
    }
    
    
    public override void Moving(int updateCount)
    {
        if(playBack != 0)
        {
            if(guidanceTime > 0)
            {
                guidanceTime--;
            }
            playBack--;
            replayGuidance.transform.position = modelPositions[Math.Min(guidanceTime, fileRowCount-1)];
        }
        else if(releaseTrigerOnce == true)
        {
            guidanceTime++;
            availableNum--;
            stopGuidance.transform.position = modelPositions[Math.Min(guidanceTime, fileRowCount-1)];
            if(availableNum <= 0)
            {
                replay = false;
                releaseTrigerOnce = false;
                replayGuidance.transform.position = new Vector3(0, 0, 0);
            }
        }
    }
    public override void GuidanceUpdate()
    {
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);
        GetControllerPos();

        if (interactUI)
        {
            trialTime += Time.deltaTime;

            if(correspondTime == -1 && guidanceTime == -1)  // すべての試行の初期動作。
            {
                //availableNum = 5;          // 初期動作でガイダンスがどれだけ進むか
                correspondTime = 0;
                guidanceTime = 0;
                trialTime = 0f;
                notAvailableNum = 0;
                replay = false;
                releaseTrigerOnce = false;
                autoMode ++;
                maxError = 0;
                updateCount = 0;
                trialFinish = false;
            }

            if(autoMode % 2 == 1)
            {
                guidanceTime++;
            }
            else
            {

                if(ErrorScore > ikiti)
                {
                    Debug.Log("ErrorScore > ikiti");
                    playBack = correspondTime - replayPointTime;
                    guidanceTime = correspondTime;
                    replay = true;
                    ErrorScore = 0;
                    maxError = 0;
                }
                else if(stopCount > 270)
                {
                    Debug.Log("stopCount > 270");
                    playBack = 90;
                    guidanceTime = correspondTime + 90;
                    replay = true;
                    ErrorScore = 0;
                    maxError = 0;
                    availableNum = 90;
                    stopCount = 0;
                }
                
                if(replay == true)
                {
                    stopUser.transform.position = RightHandPosition;
                    if(guidanceTime == -1)
                    {
                        replay = false;
                    }
                    else
                    {
                        Moving(updateCount);
                    }
                    
                }
                else if(replay == false)
                {
                    float nowError;
                    stopUser.transform.position = new Vector3(0, 0, 0);
                    stopGuidance.transform.position = new Vector3(0, 0, 0);

                    nowError = Evaluation();  // 現フレームのエラー
                    if(maxError < nowError)  // 直前のリプレイ後から、最も誤差が大きいとき
                    {
                        maxError = nowError;
                        replayPointTime = preCorrespondTime;  // replayしたときに、ここまで戻る
                        availableNum = correspondTime - preCorrespondTime;  // replay後、これだけ見本を表示してあげる。
                    }
                    ErrorScore += nowError;
                    preCorrespondTime = correspondTime;

                    updateCount++;
                }
            }

        }
        else if(guidanceTime == -1)
        {
            correspondTime = -1;  // 1試行が正常終了したことを意味する
            trialFinish = true;
        }
        else if(replay == true)
        {
            releaseTrigerOnce = true;
        }
        else    // 正常終了前に、トリガーが外れてしまった場合。
        {
            Debug.Log("トリガーが外れました");
            guidanceTime = -1;
            correspondTime = -1;
            trialFinish = true;
        }
    }

    private void GetControllerPos()
    {
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
    }
}

