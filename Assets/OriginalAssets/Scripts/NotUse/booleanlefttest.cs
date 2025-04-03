using System;
using UnityEngine;
using Valve.VR;

public class booleanlefttest : MonoBehaviour
{
    // InteractUIボタンが押されているかを判定するためのIuiという関数にSteamVR_Actions.defalt_InteractionUIを固定
    private SteamVR_Action_Boolean Iui = SteamVR_Actions.default_InteractUI;
    // 結果の格納用Boolean型変数interacrtui
    private Boolean interactUI;

    // GrabGripボタン（初期設定は側面ボタン）が押されているかを判定するためのGrabという関数にSteamVR_Actions.defalt_GrtabGripを固定
    private SteamVR_Action_Boolean GrabG = SteamVR_Actions.default_GrabGrip;
    // 結果の格納用Boolean型関数grabgrip;
    private Boolean grabGrip;

    

    // Update is called once per frame
    void Update()
    {
        // 結果をGetStateで取得してinteracrtuiに格納
        // SteamVR_Input_Sources.機器名（今回は左コントローラ）
        interactUI = Iui.GetState(SteamVR_Input_Sources.LeftHand);
        // InteractUiが押されているときにコンソールにInteractUIと表示
        if(interactUI)
        {
            Debug.Log("InteractUI");
        }

        //結果をGetStateで取得してgrapgripに格納
        //SteamVR_Input_Sources.機器名（今回は左コントローラ）
        grabGrip = GrabG.GetState(SteamVR_Input_Sources.LeftHand);
        //GrabGripが押されているときにコンソールにGrabGripと表示
        if (grabGrip)
        {
            Debug.Log("GrabGrip");
        }
        
    }
}
