from app.services.orc_parse_gemini import parse_invoice_text

if __name__ == "__main__":
    fake_ocr_text = """
    HÓA ĐƠN BÁN HÀNG
    Ngày: 15/09/2025
    Cửa hàng ABC
    Sữa tươi Vinamilk 2 hộp 50,000
    Bánh mì 1 cái 10,000
    Tổng cộng: 60,000 VND
    """
    result = parse_invoice_text(fake_ocr_text)
    print("✅ Parsed JSON:")
    print(result)
