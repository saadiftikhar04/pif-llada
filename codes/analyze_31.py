#!/usr/bin/env python
"""
Analyze Step 3.1 results and prepare for Step 3.2
"""

import json
import sys

def analyze_step31_results(json_path='pif_llada_step31_results.json'):
    """Analyze the Step 3.1 results"""
    
    print("="*80)
    print("STEP 3.1 RESULTS ANALYSIS")
    print("="*80)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n✓ Loaded results from: {json_path}")
        
        # Basic info
        print("\n" + "="*80)
        print("CONFIGURATION")
        print("="*80)
        config = data.get('config', {})
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Original vs Adversarial
        print("\n" + "="*80)
        print("PROMPT COMPARISON")
        print("="*80)
        
        original = data.get('original_prompt', 'N/A')
        adversarial = data.get('adversarial_prompt', 'N/A')
        
        print(f"\n📝 Original Prompt:")
        print(f"   {original}")
        print(f"\n🎯 Adversarial Prompt (after PiF):")
        print(f"   {adversarial}")
        
        # Show differences
        print(f"\n🔄 Changes:")
        if original != adversarial:
            orig_words = original.split()
            adv_words = adversarial.split()
            
            print(f"   Original length: {len(orig_words)} words")
            print(f"   Adversarial length: {len(adv_words)} words")
            print(f"   Word difference: {abs(len(adv_words) - len(orig_words))} words")
        else:
            print(f"   ⚠️  No changes detected!")
        
        # Responses
        print("\n" + "="*80)
        print("LLaDA RESPONSES")
        print("="*80)
        
        orig_response = data.get('original_response', 'N/A')
        adv_response = data.get('adversarial_response', 'N/A')
        
        print(f"\n📤 Original Response (baseline LLaDA):")
        print(f"   Length: {len(orig_response.split())} words")
        print(f"   Preview: {orig_response[:200]}...")
        
        print(f"\n📤 Adversarial Response (after PiF-LLaDA):")
        print(f"   Length: {len(adv_response.split())} words")
        print(f"   Preview: {adv_response[:200]}...")
        
        # Check if jailbreak
        print("\n" + "="*80)
        print("JAILBREAK ANALYSIS")
        print("="*80)
        
        orig_jailbreak = data.get('original_jailbreak', False)
        adv_jailbreak = data.get('adversarial_jailbreak', False)
        
        print(f"\n🔍 Original prompt jailbreak: {'YES ✓' if orig_jailbreak else 'NO ✗'}")
        print(f"🔍 Adversarial prompt jailbreak: {'YES ✓' if adv_jailbreak else 'NO ✗'}")
        
        if not orig_jailbreak and adv_jailbreak:
            print(f"\n🎉 SUCCESS! PiF-LLaDA achieved jailbreak!")
        elif orig_jailbreak and adv_jailbreak:
            print(f"\n⚠️  Both succeeded (original was already jailbroken)")
        elif adv_jailbreak and not orig_jailbreak:
            print(f"\n✓ PiF-LLaDA improved: jailbreak achieved")
        else:
            print(f"\n⚠️  Neither succeeded (attack did not work for this prompt)")
        
        # PiF metrics
        print("\n" + "="*80)
        print("PiF ATTACK METRICS")
        print("="*80)
        
        metrics = data.get('pif_metrics', {})
        iterations = metrics.get('iterations', 0)
        replacements = metrics.get('replacements', [])
        
        print(f"\n📊 Total iterations: {iterations}")
        print(f"📊 Total replacements: {len(replacements)}")
        
        if replacements:
            print(f"\n📝 Sample replacements (first 5):")
            for i, rep in enumerate(replacements[:5]):
                print(f"   [{rep.get('iteration', '?')}] Position {rep.get('position', '?')}: "
                      f"'{rep.get('old', '?')}' → '{rep.get('new', '?')}'")
        
        # Timing
        print("\n" + "="*80)
        print("TIMING")
        print("="*80)
        
        orig_time = data.get('original_generation_time', 0)
        adv_time = data.get('adversarial_generation_time', 0)
        pif_time = data.get('pif_attack_time', 0)
        
        print(f"\n⏱️  Original generation: {orig_time:.2f}s")
        print(f"⏱️  PiF attack: {pif_time:.2f}s")
        print(f"⏱️  Adversarial generation: {adv_time:.2f}s")
        print(f"⏱️  Total: {orig_time + pif_time + adv_time:.2f}s")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        print(f"\n✓ Step 3.1 completed successfully")
        print(f"✓ PiF made {len(replacements)} replacements in {iterations} iterations")
        print(f"✓ Original prompt: {'Jailbroken' if orig_jailbreak else 'Refused'}")
        print(f"✓ Adversarial prompt: {'Jailbroken' if adv_jailbreak else 'Refused'}")
        
        if adv_jailbreak and not orig_jailbreak:
            print(f"\n🎯 ATTACK SUCCESSFUL: PiF-LLaDA achieved jailbreak!")
        
        # Next steps
        print("\n" + "="*80)
        print("NEXT STEPS: Step 3.2")
        print("="*80)
        
        print(f"\n📋 Now ready to run Step 3.2: PiF-LLaDA on full HarmBench")
        print(f"\nTo proceed:")
        print(f"1. Review this output to ensure Step 3.1 looks correct")
        print(f"2. Run: sbatch run_pif_llada_harmbench.sh")
        print(f"3. Monitor: tail -f pif_llada_harmbench_*.out")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {json_path}")
        print(f"\nExpected output file from Step 3.1")
        print(f"Check if the job completed successfully")
        return False
    
    except json.JSONDecodeError as e:
        print(f"\n❌ Error: Invalid JSON in {json_path}")
        print(f"   {e}")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'pif_llada_step31_results.json'
    success = analyze_step31_results(json_path)
    sys.exit(0 if success else 1)