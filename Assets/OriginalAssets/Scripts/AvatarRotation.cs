using System;
using UnityEngine;
using UnityEngine.XR;
using Valve.VR;

public class AvatarRotation : MonoBehaviour
{
    private Vector3 HMDPosition;
    //HMDの回転座標格納用（クォータニオン）
    private Quaternion HMDRotationQ;
    //HMDの回転座標格納用（オイラー角）
    private Vector3 HMDRotation;
    private float initialEulerAnglesY;

    // Start is called before the first frame update
    void Start()
    {
        // 初期回転をオイラー角に変換し、y座標を取得
        initialEulerAnglesY = this.transform.rotation.eulerAngles.y;

    }

    // Update is called once per frame
    void Update()
    {
        // HMDの位置を取得する新しい方法
        InputDevice headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
        if (headDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 newHMDPosition))
        {
            HMDPosition = newHMDPosition;
        }

        //回転座標をクォータニオンで値を受け取る
        HMDRotationQ = InputTracking.GetLocalRotation(XRNode.Head);
        //取得した値をクォータニオン → オイラー角に変換
        HMDRotation = HMDRotationQ.eulerAngles;


        // HMDRotation.yについて、正面を0にし、右へ行くほど増えるよう変換
        if(HMDRotation.y > 180f)
        {
            HMDRotation.y -= 360f;
        }
        float HMDtoLine = HMDRotation.y * Mathf.PI / 90f;  // HMDtoLineの値が-2π～2πになるように調整。

        float userPosX = 2 * Mathf.Sin(HMDtoLine / 2);
        float userPosZ = 2 * Mathf.Cos(HMDtoLine / 2);


        this.transform.position = new Vector3(userPosX, 0f, userPosZ);

        // HMDtoLineを-π/2からπ/2の範囲に変換し、y軸回りの回転を設定
        float rotationY = HMDtoLine * (180f / Mathf.PI); // ラジアンを度に変換
        this.transform.rotation = Quaternion.Euler(0, initialEulerAnglesY - rotationY, 0); // y軸回りに回転
    }
}
