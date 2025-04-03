using System;
using UnityEngine;
using UnityEngine.XR;
using Valve.VR;

public class HeadMove : MonoBehaviour
{

    private Vector3 HMDPosition;
    //HMDの回転座標格納用（クォータニオン）
    private Quaternion HMDRotationQ;
    //HMDの回転座標格納用（オイラー角）
    private Vector3 HMDRotation;


    // Start is called before the first frame update
    void Start()
    {
        
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

        // Debug.Log("HeadRotation:" + HMDRotation.x + ", " + HMDRotation.y + ", " + HMDRotation.z);

        // HMDRotation.yについて、正面を0にし、右へ行くほど増えるよう変換
        if(HMDRotation.y > 180f)
        {
            HMDRotation.y -= 360f;
        }
        float HMDtoLine = HMDRotation.y * Mathf.PI / 90f;  // HMDtoLineの値が-2π～2πになるように調整。

        float userPosX;
        float userPosZ;

        userPosX = -2 * Mathf.Sin(HMDtoLine / 2);

        if((0 <= HMDtoLine && HMDtoLine < Mathf.PI) || (-2 * Mathf.PI < HMDtoLine && HMDtoLine < -Mathf.PI))
        {
            userPosZ = -2 * Mathf.Cos(HMDtoLine / 2);
        }
        else
        {
            userPosZ = -2 * Mathf.Cos(HMDtoLine / 2);
        }

        this.transform.position = new Vector3(userPosX, 0f, userPosZ);
    }
}
