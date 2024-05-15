package org.example;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import javax.net.ssl.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;

public class Main {
    public static void main(String[] args) {

        disableCertificateValidation();

        String url = "https://www.firat.edu.tr/tr/page/news/dunyanin-en-etkili-bilim-insanlari-listesinde-firat-universitesinden-42-akademisyen-yer-aldi-4391";

        try {
            Document doc = Jsoup.connect(url).get();

            Elements content = doc.select(".post-content");

            System.out.println("İçerik *************************************:" + content);


            BufferedWriter writer = null;
            try {
                writer = new BufferedWriter(new FileWriter("output.txt", true));
                for (Element element : content) {
                    writer.write(element.text());
                    writer.newLine();
                }
                writer.newLine();
                writer.newLine();
                System.out.println("HTML içeriği başarıyla output.txt dosyasına eklenmiştir.");
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (writer != null) {
                    try {
                        writer.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void disableCertificateValidation() {

        try {
            TrustManager[] trustAllCerts = new TrustManager[]{
                    new X509TrustManager() {
                        public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                            return null;
                        }

                        public void checkClientTrusted(
                                java.security.cert.X509Certificate[] certs, String authType) {
                        }

                        public void checkServerTrusted(
                                java.security.cert.X509Certificate[] certs, String authType) {
                        }
                    }
            };

            SSLContext sc = SSLContext.getInstance("TLS");
            sc.init(null, trustAllCerts, new java.security.SecureRandom());
            HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory());

            HostnameVerifier allHostsValid = (hostname, session) -> true;
            HttpsURLConnection.setDefaultHostnameVerifier(allHostsValid);
        } catch (NoSuchAlgorithmException | KeyManagementException e) {
            e.printStackTrace();
        }
    }
}
