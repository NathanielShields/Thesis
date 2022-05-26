using UnityEngine;
using UnityEngine.Android;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;


public class FingertipData : MonoBehaviour
{
    [SerializeField]
    private OVRHand ovrHand;

    [SerializeField]
    private OVRHand.Hand HandType = OVRHand.Hand.None;

    [SerializeField]
    private OVRSkeleton skeleton;

    private String path;
    private List<OVRBone> fingerBones;
    private bool pinch;

    private void OnEnable()
    {
        if(skeleton == null){
            Logger.Instance.LogError("fingerBones must be set in the inspector...");
        }
        if(ovrHand == null)
        {
            Logger.Instance.LogError("ovrHand must be set in the inspector...");
        }
        if((skeleton != null) && (ovrHand != null))
        {
            Logger.Instance.LogInfo("OvrHand and fingerBones were set correctly in the inspector... Version 35");
        }

        // Output file
        path = Application.persistentDataPath + "/" + "Q_" + DateTime.Now.ToString("MMM_dd_HH_mm_ss") + ".txt";
        //  + DateTime.Now.ToString("MMM_dd_HH_mm_ss")

        if (!File.Exists(path))
        {
            // Create a file to write to
            File.WriteAllText(path, "");
        }
        pinch = false;
    }
    
    void Start(){
        fingerBones = new List<OVRBone>(skeleton.Bones);
    }

    void Update()
    {
        // index finger pinch creates an anchor
        if(ovrHand.GetFingerIsPinching(OVRHand.HandFinger.Index))
        {
            pinch = true;
        }

        if(pinch)
        {
            if(HandType == OVRHand.Hand.HandLeft){
                Logger.Instance.Clear();
                
                if(fingerBones != null){                   
                   
                    // Timestamp to starte line
                    string line = DateTime.Now.ToString("HH:mm:ss.ffffff");
                
                    foreach(var bone in fingerBones){
                        //get the bone coordinate relative to the hand root
                        Vector3 pos = skeleton.transform.InverseTransformPoint(bone.Transform.position);
                        Logger.Instance.LogInfo($"{DateTime.Now.ToString("HH:mm:ss.ff")}, {bone.Id}, {pos.x.ToString("F6")}, {pos.y.ToString("F6")}, {pos.z.ToString("F6")}");
                        line = line + ", " + pos.x.ToString("F6") + ", " + pos.y.ToString("F6") + ", " + pos.z.ToString("F6");
                    }

                    line = line + Environment.NewLine;
                    File.AppendAllText(path, line);
                }
                else{
                    Logger.Instance.LogInfo($"No bones available");
                }
            }
        }
    }
}