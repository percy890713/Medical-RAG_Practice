import re
from typing import Union


class MedicalDeidentifier:
    """PHI 去識別化模組（三層防護）"""

    # 台灣身分證字號
    _ID_PATTERN = re.compile(r'[A-Z][12]\d{8}')
    # 台灣手機號碼
    _PHONE_PATTERN = re.compile(r'09\d{2}-?\d{3}-?\d{3}')
    # 台灣市話
    _TEL_PATTERN = re.compile(r'0[2-8]\d{1}-?\d{3,4}-?\d{4}')
    # 生日：民國/西元年月日
    _DOB_PATTERN = re.compile(
        r'(?:民國)?\d{2,3}年\d{1,2}月\d{1,2}日'
        r'|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
    )
    # 台灣地址（含市/縣 + 區/鄉/鎮 + 路/街 + 號）
    _ADDRESS_PATTERN = re.compile(
        r'[^\s，。、]{2,5}(?:市|縣)[^\s，。、]{1,4}(?:區|鄉|鎮)[^\s，。、]{1,10}(?:路|街|大道)[^\s，。、]{0,10}號[^\s，。、]{0,5}'
    )
    # 緊急聯絡人（欄位關鍵字 + 中文姓名）
    _EMERGENCY_PATTERN = re.compile(
        r'(?:緊急聯絡人|聯絡人|家屬)[：:]\s*([^\s，。、\n]{2,4})'
    )
    # 中文姓名（病患/患者/個案 後面 2-4 字，有分隔符）
    _NAME_CONTEXT_PATTERN = re.compile(
        r'(?:病患|患者|個案|姓名|病人|個管)[：:\s]+([^\s，。、\n（(【\[]{2,4})'
    )
    # 台灣常見姓氏（覆蓋約 90% 人口）
    _SURNAMES = (
        '陳林黃張李王吳劉蔡楊許鄭謝洪曾邱廖賴徐周葉'
        '蘇莊呂江何蕭羅高潘簡朱鍾彭游詹胡施沈余盧梁'
        '唐薛歐范方宋鄧杜傅侯曹魏丁石孫馬趙馮蔣韓秦'
        '尤孔嚴華金陶薑戚鄒喻柏竇章雲葛奚袁柳任俞苗'
    )
    # 中文姓名（無分隔符，但開頭須為常見姓氏）
    _NAME_CONTEXT_NO_SEP_PATTERN = re.compile(
        rf'(?:病患|患者|個案|姓名|病人|個管)([{_SURNAMES}][^\s，。、\n（(【\[{{}}]{{1,3}})'
    )

    def __init__(self):
        self._name_map: dict[str, str] = {}
        self._name_counter = 0

    def _reset_name_map(self):
        self._name_map = {}
        self._name_counter = 0

    def _get_name_placeholder(self, name: str) -> str:
        if name not in self._name_map:
            label = chr(ord('A') + self._name_counter)
            self._name_map[name] = f'[姓名_{label}]'
            self._name_counter += 1
        return self._name_map[name]

    def _replace_phi_in_text(self, text: str, consistent_names: bool = True) -> tuple[str, list[str]]:
        """對單一文字字串執行 PHI 替換，回傳 (replaced_text, detected_types)"""
        detected = []

        if self._ID_PATTERN.search(text):
            detected.append('ID_NUMBER')
        text = self._ID_PATTERN.sub('[ID_NUMBER]', text)

        if self._PHONE_PATTERN.search(text):
            detected.append('PHONE')
        text = self._PHONE_PATTERN.sub('[PHONE]', text)

        if self._TEL_PATTERN.search(text):
            if 'PHONE' not in detected:
                detected.append('PHONE')
        text = self._TEL_PATTERN.sub('[PHONE]', text)

        if self._DOB_PATTERN.search(text):
            detected.append('DOB')
        text = self._DOB_PATTERN.sub('[DOB]', text)

        if self._ADDRESS_PATTERN.search(text):
            detected.append('ADDRESS')
        text = self._ADDRESS_PATTERN.sub('[ADDRESS]', text)

        # 緊急聯絡人姓名
        def replace_emergency(m):
            name = m.group(1)
            if consistent_names:
                placeholder = self._get_name_placeholder(name)
            else:
                placeholder = '[EMERGENCY_CONTACT]'
            if 'EMERGENCY_CONTACT' not in detected:
                detected.append('EMERGENCY_CONTACT')
            return m.group(0).replace(name, placeholder)

        text = self._EMERGENCY_PATTERN.sub(replace_emergency, text)

        # 上下文姓名
        def replace_name(m):
            name = m.group(1)
            if consistent_names:
                placeholder = self._get_name_placeholder(name)
            else:
                placeholder = '[PATIENT_NAME]'
            if 'PATIENT_NAME' not in detected:
                detected.append('PATIENT_NAME')
            return m.group(0).replace(name, placeholder)

        text = self._NAME_CONTEXT_PATTERN.sub(replace_name, text)
        text = self._NAME_CONTEXT_NO_SEP_PATTERN.sub(replace_name, text)

        return text, detected

    def deidentify_record(self, record: dict) -> dict:
        """去識別化整筆病歷，回傳去識別化後的 dict"""
        self._reset_name_map()
        result = {}
        for key, value in record.items():
            if isinstance(value, str):
                cleaned, _ = self._replace_phi_in_text(value, consistent_names=True)
                result[key] = cleaned
            elif isinstance(value, list):
                result[key] = [
                    self._replace_phi_in_text(item, consistent_names=True)[0]
                    if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[key] = self.deidentify_record(value)
            else:
                result[key] = value
        return result

    def scan_query(self, query: str) -> tuple[str, list[str]]:
        """掃描使用者查詢，回傳 (cleaned_query, detected_phi_types)"""
        self._reset_name_map()
        cleaned, detected = self._replace_phi_in_text(query, consistent_names=False)
        return cleaned, detected

    def scan_output(self, text: str) -> tuple[str, bool]:
        """掃描 LLM 輸出，回傳 (text, has_phi_warning)"""
        self._reset_name_map()
        cleaned, detected = self._replace_phi_in_text(text, consistent_names=False)
        has_warning = len(detected) > 0
        return cleaned, has_warning
