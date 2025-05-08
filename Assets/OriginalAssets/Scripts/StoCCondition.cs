using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.ComponentModel;
using System;
using System.Text;
using System.Linq; // Average()などのために必要

public class StoCCondition : MonoBehaviour
{
    [Header("Model Files")]
    [SerializeField] string readFileNameA = "modelA";
    [SerializeField] string readFileNameB = "modelB";
    [SerializeField] string writeFileNameBase = "participant";

    [Header("GameObjects")]
    [SerializeField] private GameObject guidance;
    [SerializeField] private GameObject user;
    [SerializeField] GameObject startPoint, startPoint2, endPoint, endPoint2;
    [SerializeField] GameObject wristR;
    [SerializeField] Animator p_Animator;

    [Header("Settings")]
    [SerializeField] int readFileRowCountA = 1000;
    [SerializeField] int readFileRowCountB = 1000;
    [SerializeField] bool Recording = true;
    [SerializeField, Range(1, 20)] int commaPlaySpeed = 10;
    [SerializeField] bool is3PP;
    [SerializeField] float similarityThreshold = 0.9f;

    [Header("Visuals")]
    [SerializeField] AudioSource audioSource;
    [SerializeField] Material materialOdd;
    [SerializeField] Material materialEven;
    [Tooltip("If not set, relative to parent")]
    public Transform origin;

    private FileOperation fileOp;
    private AutoPlay guidanceController;

    private enum CurrentModel { ModelA, ModelB }
    private CurrentModel currentModel = CurrentModel.ModelA;
    private bool modelBSwitchTriggered = false;
    private int lastTrialCheckedForSwitch = 0;

    private Renderer betaJointsRenderer;
    private Renderer betaSurfaceRenderer;
    private int currentTrialParity = -1;
    private string betaJointsObjectName = "Beta_Joints";
    private string betaSurfaceObjectName = "Beta_Surface";

    void Start()
    {
        Transform betaJointsTransform = guidance.transform.Find(betaJointsObjectName);
        Transform betaSurfaceTransform = guidance.transform.Find(betaSurfaceObjectName);
        if (betaJointsTransform != null && betaSurfaceTransform != null)
        {
            betaJointsRenderer = betaJointsTransform.GetComponent<Renderer>();
            betaSurfaceRenderer = betaSurfaceTransform.GetComponent<Renderer>();
            if (betaJointsRenderer == null || betaSurfaceRenderer == null)
            {
                 Debug.LogError($"Renderer component not found on '{betaJointsObjectName}' or '{betaSurfaceObjectName}' under guidance.", this);
            }
        }
        else
        {
            Debug.LogError($"Child objects '{betaJointsObjectName}' or '{betaSurfaceObjectName}' not found under guidance.", this);
        }

        try
        {
            fileOp = new FileOperation(user, startPoint, startPoint2, endPoint, endPoint2);

            if (!fileOp.LoadModelData(readFileNameA, readFileRowCountA))
            {
                Debug.LogError($"Failed to load initial model A data: {readFileNameA}. Disabling component.", this);
                enabled = false;
                return;
            }
            fileOp.SetWriteFileNameBase($"{writeFileNameBase}_Simple");

            guidanceController = new AutoPlay(guidance, user, fileOp.FileRowCount, fileOp.modelPositions, fileOp.modelQuaternions,
                                       commaPlaySpeed, wristR, p_Animator);

            //fileOp.FileSettingCheck();
            if (fileOp.fileSettingWrong)
            {
                Debug.LogError("File setting check failed after initialization. Disabling component.", this);
                enabled = false;
                OnDestroy();
                UnityEditor.EditorApplication.isPlaying = false;
                return;
            }

        }
        catch (Exception ex)
        {
            Debug.LogError($"Initialization failed during Start(): {ex.Message} {ex.StackTrace}", this);
            enabled = false;
            return;
        }

        UpdateBetaJointsMaterial();
        Debug.Log($"StoCCondition Initialized. Starting with Model A ({readFileNameA}). Recording: {Recording}");
    }

    void FixedUpdate()
    {
        if (fileOp == null || guidanceController == null || !enabled) return;
        if (fileOp.fileSettingWrong)
        {
            Debug.LogError("File setting check failed after initialization. Disabling component.", this);
            enabled = false;
            OnDestroy();
            UnityEditor.EditorApplication.isPlaying = false;
            return;
        }
        guidanceController.GuidanceUpdate();

        if (audioSource != null && (guidanceController.CurrentGuidanceTime + 1) % 90 == 0 && guidanceController.CurrentGuidanceTime < fileOp.FileRowCount && guidanceController.CurrentGuidanceTime > -1)
        {
             audioSource.Play();
        }

        if (Recording)
        {
            bool wasRecordingActive = fileOp.IsRecordingActive;

            try
            {
                 fileOp.RecordingUpdate();
            }
            catch (Exception ex)
            {
                 Debug.LogError($"Error during FileOperation.RecordingUpdate(): {ex.Message} {ex.StackTrace}", this);
                 enabled = false;
                 return;
            }

            int completedTrial = fileOp.CurrentTrialCount - 1;
            if (wasRecordingActive && !fileOp.IsRecordingActive && completedTrial > 0 && completedTrial % 2 == 0)
            {
                if (currentModel == CurrentModel.ModelA && !modelBSwitchTriggered && completedTrial > lastTrialCheckedForSwitch)
                {
                    lastTrialCheckedForSwitch = completedTrial;

                    List<Vector3> userTrajectory = fileOp.GetUserTrajectoryData();

                    if (userTrajectory != null && userTrajectory.Count > 1 && fileOp.modelPositions != null && fileOp.modelPositions.Length > 0)
                    {
                        try
                        {
                            //double similarityScore = TrajectorySimilarity.CalculateDTWDotProductSimilarity(fileOp.modelPositions, userTrajectory);
                            float similarityScore = KeyFrameVectorDot(fileOp.modelPositions.ToList(), userTrajectory);
                            Debug.Log($"Trial {completedTrial} (Model A) Similarity Score: {similarityScore:F4}");

                            if (similarityScore > similarityThreshold)
                            {
                                Debug.Log($"Similarity threshold ({similarityThreshold:F4}) exceeded. Switching to Model B ({readFileNameB})...");
                                modelBSwitchTriggered = true;
                                currentModel = CurrentModel.ModelB;

                                if (!fileOp.LoadModelData(readFileNameB, readFileRowCountB))
                                {
                                    Debug.LogError($"Failed to load model B data: {readFileNameB}. Reverting to Model A.", this);
                                    currentModel = CurrentModel.ModelA;
                                    modelBSwitchTriggered = false;
                                }
                                else
                                {
                                    Debug.Log("Successfully loaded Model B data.");
                                    fileOp.SetWriteFileNameBase(writeFileNameBase);
                                    guidanceController.UpdateModelData(fileOp.modelPositions, fileOp.modelQuaternions, fileOp.FileRowCount);
                                    Debug.Log("Guidance controller updated with Model B data.");
                                }
                            }
                            else
                            {
                                Debug.Log($"Trial {completedTrial} similarity ({similarityScore:F4}) did not exceed threshold ({similarityThreshold:F4}). Continuing with Model A.");
                            }
                        }
                        catch (Exception ex)
                        {
                             Debug.LogError($"Error calculating similarity or switching model: {ex.Message} {ex.StackTrace}", this);
                        }
                    }
                    else
                    {
                         Debug.LogWarning($"Trial {completedTrial}: Not enough data for similarity calculation. User points: {(userTrajectory?.Count ?? 0)}, Model points: {(fileOp.modelPositions?.Length ?? 0)}. Skipping check.");
                    }
                }
                 else if(currentModel == CurrentModel.ModelB)
                 {
                      Debug.Log($"Completed even trial {completedTrial} using Model B.");
                 }
            }

            UpdateBetaJointsMaterial();
        }
    }

    void UpdateBetaJointsMaterial()
    {
        if (betaJointsRenderer == null || betaSurfaceRenderer == null || fileOp == null) return;

        int newParity = fileOp.CurrentTrialCount % 2;

        if (newParity != currentTrialParity)
        {
            Material targetMaterial = (newParity == 1) ? materialOdd : materialEven;

            if (targetMaterial != null)
            {
                betaJointsRenderer.material = targetMaterial;
                betaSurfaceRenderer.material = targetMaterial;
                currentTrialParity = newParity;
            }
            else
            {
                 Debug.LogWarning($"Trial {fileOp.CurrentTrialCount}: Material for {(newParity == 1 ? "Odd" : "Even")} ({(newParity == 1 ? "materialOdd" : "materialEven")}) is not assigned in the inspector.", this);
            }
        }
    }

    void OnAnimatorIK()
    {
        if (guidanceController == null || p_Animator == null || fileOp == null || fileOp.modelPositions == null || fileOp.modelQuaternions == null || fileOp.FileRowCount <= 0)
        {
             if(p_Animator != null) {
                 p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 0);
                 p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 0);
             }
             return;
        }

        int guidanceTime = guidanceController.CurrentGuidanceTime;
        int modelIndex = Math.Min(Math.Max(0, guidanceTime), fileOp.FileRowCount - 1);

        Vector3 targetPos;
        Quaternion targetRot;

        try {
             Vector3 modelPos = fileOp.modelPositions[modelIndex];
             Quaternion modelRot = fileOp.modelQuaternions[modelIndex];

            if (origin == null)
            {
                float zOffset = is3PP ? 3.0f : 0f;
                Vector3 offset = new Vector3(0, 0, zOffset);
                targetPos = modelPos + offset;
                targetRot = modelRot;
            }
            else
            {
                targetPos = origin.TransformPoint(modelPos);
                targetRot = origin.rotation * modelRot;
            }

             p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1.0f);
            p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 1.0f);
            p_Animator.SetIKPosition(AvatarIKGoal.RightHand, targetPos);
            p_Animator.SetIKRotation(AvatarIKGoal.RightHand, targetRot);
        }
        catch (IndexOutOfRangeException ex)
        {
             Debug.LogError($"OnAnimatorIK Error: Index {modelIndex} out of range for model data (Size: {fileOp.FileRowCount}). GuidanceTime: {guidanceTime}. Exception: {ex.Message}", this);
             p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 0);
             p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 0);
        }
        catch (NullReferenceException ex)
        {
            Debug.LogError($"OnAnimatorIK Error: Null reference encountered. Check if fileOp or model data is properly initialized. Exception: {ex.Message}", this);
             p_Animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 0);
             p_Animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 0);
        }

    }

    void OnDestroy()
    {
        if (fileOp != null && Recording)
        {
           Debug.Log("Closing file in OnDestroy.");
           fileOp.CloseFile();
        }
    }

    float KeyFrameVectorDot(List<Vector3> modelTraj, List<Vector3> userTraj)
    {
        List<float> dotProducts = new List<float>();
        // 軌道を差分ベクトル化
        List<Vector3> modelVec = ToDiffVectors(modelTraj);
        List<Vector3> userVec = ToDiffVectors(userTraj);

        for(int i = 0; i < 8; i++)
        {
            float modelNorm = modelVec[i].magnitude;
            float userNorm = userVec[i].magnitude;
            // ゼロベクトルでないことを確認
            if (modelNorm > float.Epsilon && userNorm > float.Epsilon)
            {
                float dotProduct = Vector3.Dot(modelVec[i], userVec[i]);
                float normalizedDotProduct = dotProduct / (modelNorm * userNorm);
                // Clamp to [-1, 1] due to potential floating point inaccuracies
                normalizedDotProduct = Math.Max(-1.0f, Math.Min(1.0f, normalizedDotProduct));
                dotProducts.Add(normalizedDotProduct);
            }
        }
        // 5. 正規化内積の平均値を返す
        if (dotProducts.Count == 0)
        {
            Debug.LogWarning("KeyFrameVectorDot: No valid dot products calculated along the path.");
            return 0.0f;
        }
        return dotProducts.Average();
        
    }

    private  List<Vector3> ToDiffVectors(List<Vector3> trajectory)
    {
        List<Vector3> vectors = new List<Vector3>();
        int[] startIndex = {0, 90, 180, 270, 360, 450, 540, 630};
        int[] endIndex = {89, 179, 269, 359, 449, 539, 629, 719};

        for(int i = 0; i < 8; i++)
        {
            vectors.Add(trajectory[endIndex[i]] - trajectory[startIndex[i]]);
        }
        return vectors;
    }
}
