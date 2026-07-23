import streamlit as st

# Assuming 'response_data' is the dictionary returned by your FastAPI backend endpoint
if response_data:
    verdict = response_data.get("verdict", {})
    dossier = response_data.get("dossier_summary", {})
    logs = response_data.get("deliberation_logs", [])
    
    st.subheader(f"📊 Rapport d'Inférence: Dossier N° {dossier.get('numero_dossier')}")
    
    # 1. High-Level Triage Status Badge
    classification = verdict.get("triage_classification", "UNKNOWN")
    if "TIER 1" in classification:
        st.error(f"⚠️ **Statut Target : {classification}**")
    else:
        st.success(f"✅ **Statut Target : {classification}**")
        
    # 2. Main Business Presentation Layer (Safe for Stakeholders)
    st.markdown("### 👔 Synthèse d'Orientation Business")
    st.info(verdict.get("business_executive_summary"))
    
    st.markdown("### 📋 Directives Opérationnelles Immédiates")
    st.warning(f"👉 {verdict.get('operational_directive')}")
    
    st.write("---")
    
    # 3. Dynamic "Show Detail" Interactive Toggle Button
    with st.expander("🔍 Afficher les détails de l'audit technique (Data Science Deep-Dive)"):
        st.markdown("### 🧠 Justification Analytique Contextuelle")
        st.write(verdict.get("technical_deep_dive"))
        
        st.markdown("### 🤖 Rapports Individuels des Agents")
        for log in logs:
            st.markdown(f"**Agent:** `{log.get('agent')}` | **Status:** `{log.get('status')}`")
            st.caption(f"*Findings:* {log.get('findings')}")
            st.json(log.get("metrics", {}))
            st.write("")
            
        st.markdown("### 📦 Metadata Système Brute (JSON)")
        st.json(response_data)