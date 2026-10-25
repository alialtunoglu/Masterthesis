"""Native dataset selection and explicit scan/publish workflow."""
import json
import re

import streamlit as st

from app.ui_utils import get_project_root
from src.datasets.dataset_registry import DatasetDefinition, DatasetRef, list_datasets
from src.datasets.dataset_registration import discover_datasets, scan_dataset, register_dataset_revision
from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime


def select_dataset(*, key, choices=None):
    root = get_project_root()
    definitions = [item for item in list_datasets(root) if item.ready]
    names = [item.dataset_id for item in definitions]
    if not names:
        st.info("Kullanıma hazır veri seti yok. Önce Veri Setleri ekranında klasörü doğrulayıp bölmeyi yayımlayın.")
        st.page_link("pages/12_Datasets.py", label="Veri seti ekle")
        st.stop()
    dataset = st.selectbox("Veri seti", names, key=f"{key}_dataset")
    definition = next(item for item in definitions if item.dataset_id == dataset)
    ref = None
    if definition.layout != "legacy":
        directory = root / "datasets/.registry" / dataset
        revisions = sorted(path.parent.name for path in directory.glob("*/manifest.json") if any((path.parent / "splits").glob("*.json")))
        revision = st.selectbox("Veri sürümü", revisions, index=revisions.index(definition.default_revision),
                                format_func=lambda value: value[:12], key=f"{key}_{dataset}_revision")
        splits = sorted(path.stem for path in (directory / revision / "splits").glob("*.json"))
        split = st.selectbox("Sabit bölme", splits, index=splits.index(definition.default_split) if definition.default_split in splits else 0,
                             format_func=lambda value: value[:12], key=f"{key}_{dataset}_{revision}_split")
        ref = DatasetRef(dataset, revision, split).to_dict()
    runtime = resolve_dataset_runtime(root, ref, dataset)
    st.caption(f"{len(runtime['class_names'])} sınıf · Sabit train / validation / test bölmesi")
    with st.expander("Sınıflar ve veri kimliği"):
        st.dataframe([{"Class ID": index, "Sınıf": name} for index, name in enumerate(runtime["class_names"])], hide_index=True)
        if ref:
            st.text(f"Revision: {ref['revision']}\nSplit: {ref['split_id']}")
        else:
            st.caption("Legacy veri protokolü: mevcut sabit split korunur; byte-hash kimliği bulunmaz.")
    return dataset, ref, runtime


def filter_result_identity(frame, *, key):
    if "dataset_ref" not in frame:
        return frame
    def identity(value):
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            return json.dumps(parsed, sort_keys=True) if isinstance(parsed, dict) else "null"
        except ValueError:
            return "null"
    identities = frame["dataset_ref"].map(identity)
    options = sorted(set(identities))
    if options == ["null"]:
        return frame
    chosen = st.selectbox("Veri sürümü / bölme", options, key=key,
                          format_func=lambda value: "Legacy (byte hash yok)" if value == "null" else json.loads(value)["revision"][:12] + " / " + json.loads(value)["split_id"][:12])
    return frame[identities == chosen]


def render_dataset_page():
    st.title("Veri Setleri")
    st.caption("Veriyi datasets klasörüne yerleştirin. Kontrol ve kayıt ham görüntüleri taşımaz veya değiştirmez.")
    root = get_project_root()
    definitions = list_datasets(root)
    st.dataframe([{"Veri seti": item.dataset_id, "Ad": item.display_name, "Klasör": item.root_relative_path,
                   "Durum": "Hazır" if item.ready else "Bölme bekliyor"} for item in definitions], hide_index=True, width="stretch")
    with st.expander("Klasör şablonu"):
        st.code("datasets/Potato/images/Healthy/*.jpg\ndatasets/Potato/images/Late_blight/*.jpg\n\n# Ya da hazır bölmeler:\ndatasets/Potato/train/<class>/*.jpg\ndatasets/Potato/val/<class>/*.jpg\ndatasets/Potato/test/<class>/*.jpg", language="text")
        st.write("En az iki sınıf gerekir. Sınıf adları korunur. Bozuk, yinelenen ve klasör dışına yönlenen görüntüler reddedilir.")
    candidates = {path.relative_to(root).as_posix() for path in discover_datasets(root)}
    candidates.update(item.root_relative_path for item in definitions if item.layout != "legacy")
    if not candidates:
        st.info("Yeni aday bulunamadı. Şablona uygun bir klasör ekleyip sayfayı yenileyin.")
        st.button("Klasörleri yenile", on_click=lambda: None)
        return
    st.subheader("Klasörü doğrula")
    candidate = st.selectbox("Veri klasörü", sorted(candidates))
    layout = st.radio("Klasör düzeni", ["imagefolder", "presplit"], horizontal=True,
                      format_func=lambda value: {"imagefolder": "images / sınıflar", "presplit": "train / val / test"}[value])
    state_key = f"dataset_scan:{candidate}:{layout}"
    if st.button("Görüntüleri kontrol et", type="primary"):
        st.session_state.pop(state_key, None)
        try:
            status = st.empty()
            def progress(count, filename):
                if count % 100 == 0:
                    status.caption(f"{count} görüntü kontrol edildi · {filename}")
            with st.spinner("Görüntüler ve içerik hash’leri doğrulanıyor..."):
                st.session_state[state_key] = scan_dataset(root, root / candidate, layout, progress)
            status.empty()
        except (ValueError, OSError) as error:
            st.error(f"Klasör doğrulanamadı: {error}. Dosyaları düzeltip yeniden kontrol edin.")
    scan = st.session_state.get(state_key)
    if scan is None:
        return
    st.success(f"{len(scan.samples)} görüntü ve {len(scan.classes)} sınıf doğrulandı.")
    st.dataframe([{"Sınıf": name, "Görüntü": sum(row['label'] == name for row in scan.samples)} for name in scan.classes], hide_index=True)
    name = candidate.split("/")[-1]
    existing = next((item for item in definitions if item.root_relative_path == candidate), None)
    dataset_id = st.text_input("Veri seti kimliği", existing.dataset_id if existing else re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_"), disabled=existing is not None)
    display_name = st.text_input("Görünen ad", existing.display_name if existing else name)
    split_seed = st.number_input("Bölme seed", 0, 2147483647, 42)
    ratios = {"train": .70, "val": .15, "test": .15}
    if layout == "imagefolder":
        columns = st.columns(3)
        for column, part, label in zip(columns, ratios, ["Train (%)", "Validation (%)", "Test (%)"]):
            ratios[part] = column.number_input(label, 1.0, 98.0, ratios[part] * 100) / 100
    else:
        st.caption("Mevcut train / val / test atamaları aynen korunur.")
    if st.button("Kaydet ve sabit bölmeyi yayımla", disabled=abs(sum(ratios.values()) - 1) > 1e-8):
        try:
            with st.spinner("Güncellik kontrol edilip değişmez kayıt yayımlanıyor..."):
                revision = register_dataset_revision(root, DatasetDefinition(dataset_id, display_name, candidate, layout), scan)
                ref = create_registered_split(root, revision.dataset_id, revision.revision, ratios, int(split_seed))
            st.success(f"{dataset_id} eğitim için hazır. Sürüm {ref.revision[:12]} · Bölme {ref.split_id[:12]}")
            st.page_link("pages/0_New_Experiment.py", label="Bu veriyle yeni deney oluştur")
        except (ValueError, OSError) as error:
            st.error(f"Yayımlanamadı: {error}. Önceki hazır sürüm değiştirilmedi; hatayı düzeltip tekrar deneyin.")
