using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;
using Valve.VR;

public class AutoPlay : BaseGuidance // ガイダンスに関する計算・処理を行う。
{
    //private int correspondTime = 0;  // Userの現在地に対応するModelの時間。 値が-1のとき、試行と試行の間であることを意味する
    private int guidanceTime = 0;   // ガイダンスの現在の時間。値が-1のとき、ユーザーが右端まで到達したことを意味する
    public int GuidanceTime
    {
        get {return guidanceTime;}
    }

    private float playSpeed = 1f;
    private float forSpeedChange = 0f;

    // InteractUIボタンが押されているかを判定するためのIuiという関数にSteamVR_Actions.defalt_InteractionUIを固定
    private SteamVR_Action_Boolean Iui = SteamVR_Actions.default_InteractUI;
    // 結果の格納用Boolean型変数interacrtui
    private Boolean interactUI;

    // GrabGripボタン（初期設定は側面ボタン）が押されているかを判定するためのGrabという関数にSteamVR_Actions.defalt_GrtabGripを固定
    private SteamVR_Action_Boolean GrabG = SteamVR_Actions.default_GrabGrip;
    // 結果の格納用Boolean型関数grabgrip;
    private Boolean grabGrip;
    private GameObject wristR;
    private Animator p_Animator;


    public AutoPlay(GameObject guidance, GameObject user, int fileRowCount, Vector3[] positions, Quaternion[] quaternions,
     int commaPlaySpeed, Material[] materialArray, GameObject wristR, Animator p_Animator)
        : base(guidance, user, fileRowCount, positions, quaternions, materialArray)
    {
        this.playSpeed = (float)commaPlaySpeed/10f;
        this.wristR = wristR;
        this.p_Animator = p_Animator;
    }

    public override float Evaluation()
    {
        // 呼び出されない
        return -1f;
    }

    public override void Moving(int updateCount)
    {
        // 呼び出されない
    }
    public override void GuidanceUpdate()
    {
        // 結果をGetStateで取得してinteracrtuiに格納
        // SteamVR_Input_Sources.機器名（今回は左コントローラ）
        // トリガーを押したらinteractUIがtrue
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);
        if (interactUI)
        {
            if(guidanceTime < fileRowCount)
            {
                
                // wristR.transform.position = modelPositions[Math.Min(guidanceTime, fileRowCount - 1)];
                // wristR.transform.rotation = Quaternion.Euler(-90f, -90f, 0f) * modelQuaternions[Math.Min(guidanceTime, fileRowCount - 1)];
                // Vector3 elbowDir = wristR.transform.rotation * Vector3.up;
                // Vector3 elbowVec = new Vector3(-0.27f * elbowDir.x, -0.27f * elbowDir.y, -0.27f * elbowDir.z);
                // elbowR.transform.position = modelPositions[Math.Min(guidanceTime, fileRowCount - 1)] + elbowVec;
                // elbowR.transform.rotation = Quaternion.Euler(-90f, -90f, 0f) * modelQuaternions[Math.Min(guidanceTime, fileRowCount - 1)];
                // elbowR.transform.LookAt(wristR.transform.position, Vector3.right);
                // elbowR.transform.rotation = Quaternion.Euler(90f, 90f, 180f) * elbowR.transform.rotation;
                // shoulderR.transform.LookAt(Quaternion.Euler(-90f, -90f, 0f) * elbowR.transform.position, Vector3.up);
                

                //guidance.transform.position += modelPositions[Math.Min(guidanceTime, fileRowCount - 1)] - wristR.transform.position;
                //guidance.transform.rotation *= Quaternion.Inverse(wristR.transform.rotation) * modelQuaternions[Math.Min(guidanceTime, fileRowCount - 1)];
                forSpeedChange += playSpeed;
                if(forSpeedChange >= 1.0f)
                { 
                    guidanceTime += (int)forSpeedChange;
                    forSpeedChange -= (int)forSpeedChange;
                }
            }
        }
        else
        {
            guidanceTime = 0;
        }
    }

}