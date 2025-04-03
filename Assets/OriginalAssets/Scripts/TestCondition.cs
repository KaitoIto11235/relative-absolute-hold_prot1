using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;
using Valve.VR;

public class TestCondition : MonoBehaviour
{
    [SerializeField] AudioSource audioSource;
    [SerializeField] private GameObject user;
    [SerializeField] string readFileName = "default";
    [SerializeField] string writeFileName = "default";
    [SerializeField] int readFileRowCount = 1000;
    [SerializeField] GameObject startPoint, endPoint, startPoint2, endPoint2;
    FileOperation testFile;
    private int timer = 0;
    private SteamVR_Action_Boolean Iui = SteamVR_Actions.default_InteractUI;
    private Boolean interactUI = false;

    void Start()
    {
        testFile = new FileOperation(readFileName, readFileRowCount, writeFileName, user,
         startPoint, startPoint2, endPoint, endPoint2);
        testFile.WriteOpenData();
        testFile.ReadOpenData();
        testFile.FileSettingCheck();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);
        if(interactUI)
        {
            timer++;
        }
        else
        {
            timer = 0;
        }
        // 1秒に1度、効果音を鳴らす
        if(timer != 0 && timer % 90 == 0 && timer < 721)
        {
            audioSource.Play();
        }
        
        testFile.RecordingUpdate();
    }
}
