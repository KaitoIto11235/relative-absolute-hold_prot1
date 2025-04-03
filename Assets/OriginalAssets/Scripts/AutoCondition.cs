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

    void Start()
    {
        if(Recording)
        {
            autoFile = new FileOperation(readFileName, readFileRowCount, writeFileName, user,
             startPoint, startPoint2, endPoint, endPoint2);
            autoFile.WriteOpenData();
        }
        else
        {
            autoFile = new FileOperation(readFileName, readFileRowCount, user, 
             startPoint, startPoint2, endPoint, endPoint2);
        }
        autoGuidance = new AutoPlay(guidance, user, readFileRowCount, autoFile.modelPositions, autoFile.modelQuaternions,
         commaPlaySpeed, materialArray, wristR, p_Animator);
        autoFile.ReadOpenData();

        
        autoFile.FileSettingCheck();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        autoGuidance.GuidanceUpdate();
        
        // 1秒ごとに、効果音を鳴らす
        if((autoGuidance.GuidanceTime+1) % 90 == 0 && autoGuidance.GuidanceTime < 721 && autoGuidance.GuidanceTime > -1)
        {
            audioSource.Play();
        }

        if(Recording)
        {
            autoFile.RecordingUpdate();
        }
    }

    void OnAnimatorIK()
    {
        Vector3 TargetPos = new Vector3(0, 0, 0);
        Quaternion TargetRot = new Quaternion(0, 0, 0, 0);
        if(origin == null)
        {
            float three = 0;
            if(is3PP == true)
            {
                three = 3;
            }
            Vector3 offset = new Vector3(0, 0, three);
            TargetPos = autoGuidance.ModelPositions[Math.Min(autoGuidance.GuidanceTime, readFileRowCount - 1)] + offset;
            TargetRot = autoGuidance.ModelQuaternions[Math.Min(autoGuidance.GuidanceTime, readFileRowCount - 1)];
        }
        else
        {
            TargetPos = origin.TransformPoint(autoGuidance.ModelPositions[Math.Min(autoGuidance.GuidanceTime, readFileRowCount - 1)]);
            TargetRot = origin.rotation * autoGuidance.ModelQuaternions[Math.Min(autoGuidance.GuidanceTime, readFileRowCount - 1)];

        }
        
        // 右手のIKを有効化する(重み:1.0)
        p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1.0f);
        p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 1.0f);

        // 右手のIKのターゲットを設定する
        // p_Animator.SetIKPosition(AvatarIKGoal.RightHand, autoGuidance.ModelPositions[Math.Min(autoGuidance.GuidanceTime, readFileRowCount - 1)]);
        // p_Animator.SetIKRotation(AvatarIKGoal.RightHand, autoGuidance.ModelQuaternions[Math.Min(autoGuidance.GuidanceTime, readFileRowCount - 1)]);

        p_Animator.SetIKPosition(AvatarIKGoal.RightHand, TargetPos);
        p_Animator.SetIKRotation(AvatarIKGoal.RightHand, TargetRot);
    }
}


