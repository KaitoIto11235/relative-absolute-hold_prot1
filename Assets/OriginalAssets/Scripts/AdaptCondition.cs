using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;

public class AdaptCondition : MonoBehaviour
{
    [SerializeField] AudioSource audioSource;
    [SerializeField] private GameObject guidance, user;
    [SerializeField] string readFileName = "default";
    [SerializeField] string writeFileName = "default";
    [SerializeField] [Range(1, 6)] int experiment4_condition = 7;
    [SerializeField] GameObject startPoint, endPoint, startPoint2, endPoint2;

    [SerializeField] int readFileRowCount = 1000;
    FileOperation adaptFile;
    //AdaptRePlay adaptGuidance;
    AdaptPlay adaptGuidance;
    [SerializeField] bool Recording = false;

    [SerializeField] Material[] materialArray = new Material[3];
    // User停止時に手の上に表示されるオブジェクト
    [SerializeField] GameObject stopUser, stopGuidance, replayGuidance;
    [SerializeField] GameObject wristR;
    [SerializeField] Animator p_Animator;
    int audioCount = 1;

    int recordingCount = 0;

    
    void Start()
    {
        if(Recording)
        {
            adaptFile = new FileOperation(readFileName, readFileRowCount, writeFileName, user,
             startPoint, startPoint2, endPoint, endPoint2);
            adaptFile.WriteOpenData();
        }
        else
        {
            adaptFile = new FileOperation(readFileName, readFileRowCount, user,
             startPoint, startPoint2, endPoint, endPoint2);
        }
        adaptGuidance = new AdaptPlay(guidance, user, readFileRowCount, adaptFile.modelPositions, adaptFile.modelQuaternions, materialArray,
         experiment4_condition, stopUser, stopGuidance, replayGuidance, wristR);
        adaptFile.ReadOpenData();

        adaptFile.FileSettingCheck();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        
        adaptGuidance.GuidanceUpdate();

        // 1秒に1度、効果音を鳴らす
        if((adaptGuidance.GuidanceTime+1) % 90 == 0 && (adaptGuidance.GuidanceTime < 721) && (adaptGuidance.GuidanceTime > -1))
        {
            audioSource.Play();
        }
        // if(adaptGuidance.TrialFinish == true)
        // {
        //     audioCount = 0;
        // }
        // if((adaptGuidance.CorrespondTime > 90 * audioCount && adaptGuidance.AutoMode % 2 == 0) 
        // || (adaptGuidance.GuidanceTime > 90 * audioCount && adaptGuidance.AutoMode % 2 == 1))
        // {
        //     audioSource.Play();
        //     audioCount++;
        // }

        if(Recording)
        {
            if(adaptFile.InteractUI == true)
            {
                recordingCount++;
            }
            else
            {
                recordingCount = 1;
            }
            adaptFile.RecordingUpdate(adaptGuidance.DistToFile, adaptGuidance.UserLevel, adaptGuidance.TrialOffset, adaptGuidance.LevelOffset);
            //↓AdaptRePlay用
            // adaptFile.RecordingUpdate(adaptGuidance.PreAccuracyMap[Math.Min(recordingCount - 1, readFileRowCount)], adaptGuidance.CorrespondTime, 
            // adaptGuidance.TrialOffset, adaptGuidance.LevelOffset);
            //↑ここまで
        }
    }
    void OnAnimatorIK()
    {
        // 右手のIKを有効化する(重み:1.0)
        p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1.0f);
        p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 1.0f);

        // 右手のIKのターゲットを設定する
        
        if(adaptGuidance.GuidanceTime > 0)
        {
            p_Animator.SetIKPosition(AvatarIKGoal.RightHand, adaptGuidance.ModelPositions[Math.Min(adaptGuidance.GuidanceTime, readFileRowCount - 1)]);
            p_Animator.SetIKRotation(AvatarIKGoal.RightHand, adaptGuidance.ModelQuaternions[Math.Min(adaptGuidance.GuidanceTime, readFileRowCount - 1)]);
        }
        else
        {
            p_Animator.SetIKPosition(AvatarIKGoal.RightHand, adaptGuidance.ModelPositions[0]);
            p_Animator.SetIKRotation(AvatarIKGoal.RightHand, adaptGuidance.ModelQuaternions[0]);
        }

        // if(adaptGuidance.CorrespondTime >= readFileRowCount - 10 && adaptGuidance.GuidanceTime != -1)
        // {
        //     p_Animator.SetIKPosition(AvatarIKGoal.RightHand, adaptGuidance.ModelPositions[0]);
        //     p_Animator.SetIKRotation(AvatarIKGoal.RightHand, adaptGuidance.ModelQuaternions[0]);
        // }
        // AdaptRePlay用
        if(adaptGuidance.GuidanceTime >= readFileRowCount - 10 && adaptGuidance.GuidanceTime != -1)
        {
            p_Animator.SetIKPosition(AvatarIKGoal.RightHand, adaptGuidance.ModelPositions[0]);
            p_Animator.SetIKRotation(AvatarIKGoal.RightHand, adaptGuidance.ModelQuaternions[0]);
        }

        if(!adaptGuidance.InteractUI && adaptGuidance.GuidanceTime != -1)
        {
            p_Animator.SetIKPosition(AvatarIKGoal.RightHand, adaptGuidance.ModelPositions[0]);
            p_Animator.SetIKRotation(AvatarIKGoal.RightHand, adaptGuidance.ModelQuaternions[0]);
        }
    }
}
