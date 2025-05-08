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
        // FileOperation の初期化方法を変更
        try
        {
            // 1. 新しい5引数コンストラクタでインスタンス作成
            testFile = new FileOperation(user, startPoint, startPoint2, endPoint, endPoint2);

            // 2. モデルデータを読み込む (ファイル名が指定されている場合)
            if (!string.IsNullOrEmpty(readFileName))
            {
                if (!testFile.LoadModelData(readFileName, readFileRowCount))
                {
                    Debug.LogError($"TestCondition: Failed to load model data '{readFileName}'. Disabling component.", this);
                    enabled = false;
                    return;
                }
            }
            else
            {
                Debug.LogWarning("TestCondition: readFileName is not set. Model data will not be loaded.", this);
            }

            // 3. 書き込みファイル名を設定 (ファイル名が指定されている場合)
            if (!string.IsNullOrEmpty(writeFileName))
            {
               testFile.SetWriteFileNameBase(writeFileName);
            }
            else
            {

            Debug.LogError("TestCondition: writeFileName is not set. Recording will likely fail.", this);
            }

            Debug.Log("TestCondition initialized successfully.");
        }
        catch (Exception ex)
        {
            Debug.LogError($"TestCondition: Initialization failed. Error: {ex.Message} {ex.StackTrace}", this);
            enabled = false;
        }

        // WriteOpenData() と ReadOpenData() の呼び出しは不要になったため削除
        // testFile.WriteOpenData();
        // testFile.ReadOpenData();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (testFile.fileSettingWrong)
        {
            Debug.LogError("File setting check failed after initialization. Disabling component.", this);
            enabled = false;
            UnityEditor.EditorApplication.isPlaying = false;
            return;
        }
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
