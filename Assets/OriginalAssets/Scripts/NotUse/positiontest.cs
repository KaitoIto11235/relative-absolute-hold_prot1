using System;
using UnityEngine;
using UnityEngine.XR;
using Valve.VR;

public class positiontest : MonoBehaviour
{
    //HMDの位置座標格納用
    private Vector3 HMDPosition;
    //HMDの回転座標格納用（クォータニオン）
    private Quaternion HMDRotationQ;
    //HMDの回転座標格納用（オイラー角）
    private Vector3 HMDRotation;

    //左コントローラの位置座標格納用
    private Vector3 LeftHandPosition;
    //左コントローラの回転座標格納用（クォータニオン）
    private Quaternion LeftHandRotationQ;
    //左コントローラの回転座標格納用
    private Vector3 LeftHandRotation;

    //右コントローラの位置座標格納用
    private Vector3 RightHandPosition;
    //右コントローラの回転座標格納用（クォータニオン）
    private Quaternion RightHandRotationQ;
    //右コントローラの回転座標格納用
    private Vector3 RightHandRotation;

    //1フレーム毎に呼び出されるUpdateメゾット
    void Update()
    {

        // HMDの位置を取得する新しい方法
        InputDevice headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
        if (headDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 newHMDPosition))
        {
            HMDPosition = newHMDPosition;
        }

        // 左コントローラの位置を取得する新しい方法
        InputDevice leftHandDevice = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        if (leftHandDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 newLeftHandPosition))
        {
            LeftHandPosition = newLeftHandPosition;
        }

        // 右コントローラの位置を取得する新しい方法
        InputDevice rightHandDevice = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (rightHandDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 newRightHandPosition))
        {
            RightHandPosition = newRightHandPosition;
        }


        //取得したデータを表示（HMDP：HMD位置，HMDR：HMD回転，LFHR：左コン位置，LFHR：左コン回転，RGHP：右コン位置，RGHR：右コン回転）
        // Debug.Log("HMDP:" + HMDPosition.x + ", " + HMDPosition.y + ", " + HMDPosition.z);
        // Debug.Log("LFHP:" + LeftHandPosition.x + ", " + LeftHandPosition.y + ", " + LeftHandPosition.z);
        Debug.Log("RGHP:" + RightHandPosition.x + ", " + RightHandPosition.y + ", " + RightHandPosition.z);
    }
}
