using UnityEngine;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Text;
using UnityEngine.XR;
using Valve.VR;

public class FileOperation  // ファイルの読み書きを行う。
{
    private GameObject recordObject, startPoint, startPoint2, endPoint, endPoint2;
    private bool readFileOpenFlag = false;
    private bool writeFileOpenFlag = false;
    private string readFileName;
    private int fileRowCount;
    private string writeFileName;
    private StreamReader sr;
    private StreamWriter sw;
    public Vector3[] modelPositions;
    public Quaternion[] modelQuaternions;
    // public Vector3[] userPositions;

    // recording用変数↓
	private int trialCount = 1;
    public int TrialCount
    { 
        get{ return trialCount; }
    }
    private bool startFlag = false; // true: 準備OK
    private float time = 0f;        // true: どこかでファイル設定が間違っていて開けないため、再生停止

    private bool fileSettingWrong = false;
    private bool testFlag = false;


    // InteractUIボタンが押されているかを判定するためのIuiという関数にSteamVR_Actions.defalt_InteractionUIを固定
    private SteamVR_Action_Boolean Iui = SteamVR_Actions.default_InteractUI;
    // 結果の格納用Boolean型変数interacrtui
    private Boolean interactUI;
    public Boolean InteractUI => interactUI;

    // GrabGripボタン（初期設定は側面ボタン）が押されているかを判定するためのGrabという関数にSteamVR_Actions.defalt_GrtabGripを固定
    private SteamVR_Action_Boolean GrabG = SteamVR_Actions.default_GrabGrip;
    // 結果の格納用Boolean型関数grabgrip;
    private Boolean grabGrip;
    private Vector3 RightHandPosition;
    private Quaternion RightHandRotationQ;

    // 書き込みなしコンストラクタ
    public FileOperation(string readFileName, int fileRowCount, GameObject recordObj,
     GameObject startPoint, GameObject startPoint2, GameObject endPoint, GameObject endPoint2)
    {
        this.readFileName = readFileName;
        this.fileRowCount = fileRowCount;
        this.recordObject = recordObj;
        modelPositions = new Vector3[fileRowCount];
        modelQuaternions = new Quaternion[fileRowCount];
        this.startPoint = startPoint;
        this.startPoint2 = startPoint2;
        this.endPoint = endPoint;
        this.endPoint2 = endPoint2;
        // userPositions = new Vector3[fileRowCount];
    }

    // 読み込みなしコンストラクタ
    public FileOperation(string writeFileName, GameObject recordObj,
     GameObject startPoint, GameObject startPoint2, GameObject endPoint, GameObject endPoint2)
    {
        this.writeFileName = writeFileName;
        this.recordObject = recordObj;
        this.startPoint = startPoint;
        this.startPoint2 = startPoint2;
        this.endPoint = endPoint;
        this.endPoint2 = endPoint2;
        testFlag = true;
    }

    // 読み書きありコンストラクタ
    public FileOperation(string readFileName, int fileRowCount, string writeFileName, GameObject recordObj,
     GameObject startPoint, GameObject startPoint2, GameObject endPoint, GameObject endPoint2)
    {
        this.recordObject = recordObj;
        this.startPoint = startPoint;
        this.startPoint2 = startPoint2;
        this.endPoint = endPoint;
        this.endPoint2 = endPoint2;
        this.readFileName = readFileName;
        this.writeFileName = writeFileName;
        this.fileRowCount = fileRowCount;
        modelPositions = new Vector3[fileRowCount];
        modelQuaternions= new Quaternion[fileRowCount];
        // userPositions = new Vector3[fileRowCount];
    }
    public void ReadOpenData()
    {
        if (!readFileOpenFlag)
        {
            string file;
            {
                //file = Application.persistentDataPath + FileName + ".csv";

                file = Application.dataPath + @"/originalAssets/File/" + readFileName + ".csv";
            }

            if(File.Exists(file))
            {
                sr = new StreamReader(new FileStream(file, FileMode.Open), Encoding.UTF8);
                int i = 0;
                while(sr.EndOfStream == false)
                {
                    string line = sr.ReadLine();
                    string[] values = line.Split(',');
                    if (values.Length >= 5 && i >= 1)
                    {
                        float PosX = float.Parse(values[2]);
                        float PosY = float.Parse(values[3]);
                        float PosZ = float.Parse(values[4]);
                        Vector3 position = new Vector3(PosX, PosY, PosZ);

                        float QuaX = float.Parse(values[5]);
                        float QuaY = float.Parse(values[6]);
                        float QuaZ = float.Parse(values[7]);
                        float QuaW = float.Parse(values[8]);
                        Quaternion quaternion = new Quaternion(QuaX, QuaY, QuaZ, QuaW);
                        // ここでVector3を使用するか、配列に保存する
                        modelPositions[i-1] = position;
                        modelQuaternions[i-1] = quaternion;
                    }
                    i++;
                }
                Vector3 start = modelPositions[0];
                Vector3 end = modelPositions[i-2];
                startPoint.transform.position = modelPositions[0];
                startPoint2.transform.position = new Vector3(start.x, start.y, start.z - 3f);
                endPoint.transform.position = modelPositions[i-2];
                endPoint2.transform.position = new Vector3(end.x, end.y, end.z - 3f);


                if(i-1 != fileRowCount)
                {
                    Debug.Log("FileRowCountが不適切です。\n" + (i - 1) + "に設定してください。" + readFileName);
                    fileSettingWrong = true;
                }

                readFileOpenFlag = true;
                Debug.Log(file);

                ReadCloseData();
            }

            if (!File.Exists(file))
            {
                Debug.Log("ファイルが見つかりません。");
            }
        }
        else
        {
            Debug.Log("File has already opened.");
        }
    }

    public StreamWriter WriteOpenData()
    {
        if (!writeFileOpenFlag)
        {
            string file = Application.dataPath + @"/originalAssets/File/" + writeFileName + ".csv";

            if (!File.Exists(file))
            {
                sw = File.CreateText(file);
                sw.Flush();
                sw.Dispose();

                //UTF-8で生成...2番目の引数はtrueで末尾に追記、falseでファイルごとに上書き
                sw = new StreamWriter(new FileStream(file, FileMode.Open), Encoding.UTF8);

                string[] s1 =
                {
                "Trial", "time",
                "PositionX", "PositionY", "PositionZ",
                "RotationQX", "RotationQY", "RotationQZ", "RotationQW",
                "frameDiff", "userLevel", "preTrialOffset", "preLevelOffset",
                };
                string s2 = string.Join(",", s1);
                sw.WriteLine(s2);
                sw.Flush();

                writeFileOpenFlag = true;
                Debug.Log("Create_csv");
                Debug.Log(file);

                // startPoint.GetComponent<MeshRenderer>().material.color = Color.yellow;  // 記録されていることを示すために、startPointを黄色にする。

                return sw;
            }
            else
            {
                Debug.Log("そのファイルは既に存在しています。ファイル名をInspectorから変更してください。"); 
                fileSettingWrong = true;
                return null;
            }
        }
        else
        {
            Debug.Log("ファイルは既に開かれています。");
            return null;
        }
    }

    public void FileSettingCheck()
    {
        if(fileSettingWrong)
        {
            UnityEditor.EditorApplication.isPlaying = false;            // エディタの再生を強制終了
        }
    }
    public void RecordingUpdate()
    {
        // 結果をGetStateで取得してinteracrtuiに格納
        // SteamVR_Input_Sources.機器名（今回は左コントローラ）
        // トリガーを押したらinteractUIがtrue
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);

        //結果をGetStateで取得してgrapgripに格納
        //SteamVR_Input_Sources.機器名（今回は左コントローラ）
        // 側面ボタンを押したらgrabGripがtrue
        grabGrip = GrabG.GetState(SteamVR_Input_Sources.RightHand);


        time += Time.deltaTime;

        // 右コントローラの姿勢を取得する方法
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
        //RightHandRotationQ = Quaternion.AngleAxis(90, recordObject.transform.forward ) * RightHandRotationQ;
        Vector3 euler = RightHandRotationQ.eulerAngles;
        euler.z -= 90; // z軸回りに-90度回転
        RightHandRotationQ = Quaternion.Euler(euler);
        
        if (interactUI && !startFlag) // startPoint付近に入ったとき
        {
            time = 0f;
            startFlag = true;
            // startPoint.GetComponent<MeshRenderer>().material.color = Color.white;
            // endPoint.GetComponent<MeshRenderer>().material.color = Color.green;
            // startPoint2.GetComponent<MeshRenderer>().material.color = Color.white;
            // endPoint2.GetComponent<MeshRenderer>().material.color = Color.green;
        }
        else if (interactUI && startFlag)
        {
            // 記録中のフレームでデータを保存
            SaveData(RightHandPosition, RightHandRotationQ);
        }

        // if (interactUI && startFlag && Vector3.Distance(RightHandPosition, endPoint.transform.position) < 0.1f) // endPoint付近に入ったとき
        // {
            //endPoint.GetComponent<MeshRenderer>().material.color = Color.white;
        //     EndData();
        //     startFlag = false;
        // }
        if (!interactUI && startFlag) // endPoint付近に入ったとき
        {
            // endPoint.GetComponent<MeshRenderer>().material.color = Color.white;
            EndData();
            startFlag = false;
            // startPoint.GetComponent<MeshRenderer>().material.color = Color.yellow;
            // endPoint.GetComponent<MeshRenderer>().material.color = Color.white;
            // startPoint2.GetComponent<MeshRenderer>().material.color = Color.yellow;
            // endPoint2.GetComponent<MeshRenderer>().material.color = Color.white;
        }

        if (Input.GetKeyDown(KeyCode.Return)) // エンターキーが押されたら、ファイルを閉じる。
        {
            WriteCloseData();
        }

    }
    public void RecordingUpdate(float distToFile, int userLevel, int trialOffset, int levelOffset)
    {
        time += Time.deltaTime;

        // 結果をGetStateで取得してinteracrtuiに格納
        // SteamVR_Input_Sources.機器名（今回は左コントローラ）
        // トリガーを押したらinteractUIがtrue
        interactUI = Iui.GetState(SteamVR_Input_Sources.RightHand);
        //結果をGetStateで取得してgrapgripに格納
        //SteamVR_Input_Sources.機器名（今回は左コントローラ）
        // 側面ボタンを押したらgrabGripがtrue
        grabGrip = GrabG.GetState(SteamVR_Input_Sources.RightHand);


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

        if (interactUI && !startFlag) // トリガーを押した時
        {
            time = 0f;
            startFlag = true;
            // startPoint.GetComponent<MeshRenderer>().material.color = Color.white;
            // endPoint.GetComponent<MeshRenderer>().material.color = Color.green;
            // startPoint2.GetComponent<MeshRenderer>().material.color = Color.white;
            // endPoint2.GetComponent<MeshRenderer>().material.color = Color.green;
        }
        else if (interactUI && startFlag)
        {
            // 記録中のフレームでデータを保存
            SaveData(RightHandPosition, RightHandRotationQ, distToFile, userLevel, trialOffset, levelOffset);
        }

        // if (interactUI && startFlag && Vector3.Distance(RightHandPosition, endPoint.transform.position) < 0.1f) // endPoint付近に入ったとき
        // {
        //     endPoint.GetComponent<MeshRenderer>().material.color = Color.white;
        //     EndData();
        //     startFlag = false;
        // }
        if (!interactUI && startFlag) // トリガーを外した時
        {
            //endPoint.GetComponent<MeshRenderer>().material.color = Color.white;
            EndData();
            startFlag = false;

            // startPoint.GetComponent<MeshRenderer>().material.color = Color.yellow;
            // endPoint.GetComponent<MeshRenderer>().material.color = Color.white;
            // startPoint2.GetComponent<MeshRenderer>().material.color = Color.yellow;
            // endPoint2.GetComponent<MeshRenderer>().material.color = Color.white;
        }

        if (Input.GetKeyDown(KeyCode.Return)) // エンターキーが押されたら、ファイルを閉じる。
        {
            WriteCloseData();
        }

    }

    public void EndData()
    {
        sw.WriteLine("");
        sw.Flush();
        Debug.Log("End_Trail:" + trialCount);
        trialCount++;
    }

    public void SaveData(Vector3 RightHandPosition, Quaternion RightHandRotationQ)
    {
        string[] s1 =
        {
            Convert.ToString(trialCount), Convert.ToString(time),
            Convert.ToString(RightHandPosition.x), Convert.ToString(RightHandPosition.y), Convert.ToString(RightHandPosition.z),
            Convert.ToString(RightHandRotationQ.x), Convert.ToString(RightHandRotationQ.y), Convert.ToString(RightHandRotationQ.z),
            Convert.ToString(RightHandRotationQ.w),
        };
        string[] s2 =
        {
            "test" + Convert.ToString(trialCount), Convert.ToString(time),
            Convert.ToString(RightHandPosition.x), Convert.ToString(RightHandPosition.y), Convert.ToString(RightHandPosition.z),
            Convert.ToString(RightHandRotationQ.x), Convert.ToString(RightHandRotationQ.y), Convert.ToString(RightHandRotationQ.z),
            Convert.ToString(RightHandRotationQ.w),
        };
        
        string s3 = string.Join(",", s1);
        string s4 = string.Join(",", s2);
        if(!testFlag)
        {
            sw.WriteLine(s3);
        }
        else
        {
            sw.WriteLine(s4);
        }
        
        sw.Flush();
    }
    public void SaveData(Vector3 RightHandPosition, Quaternion RightHandRotationQ, float distToFile, int userLevel, int trialOffset, int levelOffset)
    {
        string[] s1 =
        {
            Convert.ToString(trialCount), Convert.ToString(time),
            Convert.ToString(RightHandPosition.x), Convert.ToString(RightHandPosition.y), Convert.ToString(RightHandPosition.z),
            Convert.ToString(RightHandRotationQ.x), Convert.ToString(RightHandRotationQ.y), Convert.ToString(RightHandRotationQ.z),
            Convert.ToString(RightHandRotationQ.w),
            Convert.ToString(distToFile), Convert.ToString(userLevel), Convert.ToString(trialOffset), Convert.ToString(levelOffset),
        };
        string[] s2 =
        {
            "test" + Convert.ToString(trialCount), Convert.ToString(time),
            Convert.ToString(RightHandPosition.x), Convert.ToString(RightHandPosition.y), Convert.ToString(RightHandPosition.z),
            Convert.ToString(RightHandRotationQ.x), Convert.ToString(RightHandRotationQ.y), Convert.ToString(RightHandRotationQ.z),
            Convert.ToString(RightHandRotationQ.w),
            Convert.ToString(distToFile), Convert.ToString(userLevel), Convert.ToString(trialOffset), Convert.ToString(levelOffset),
        };
        
        string s3 = string.Join(",", s1);
        string s4 = string.Join(",", s2);
        if(!testFlag)
        {
            sw.WriteLine(s3);
        }
        else
        {
            sw.WriteLine(s4);
        }
        
        sw.Flush();
    }
    private void ReadCloseData()
    {
        if (sr != null) // swがnullでないことを確認
        {
            sr.Dispose();
            Debug.Log("CloseRead_csv");
            readFileOpenFlag = false;
        }
        else
        {
            Debug.Log("StreamReaderが初期化されていません。");
        }
    }
    private void WriteCloseData()
    {
        if (sw != null) // swがnullでないことを確認
        {
            sw.Dispose();
            Debug.Log("CloseWrite_csv");
            writeFileOpenFlag = false;  
        }
        else
        {
            Debug.Log("StreamWriterが初期化されていません。");
        }
    }
}
