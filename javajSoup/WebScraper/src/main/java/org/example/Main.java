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
        https://www.firat.edu.tr/tr/page/news?page=2
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter("duyuru.txt", true));

            for (int page = 35; page <= 40; page++) { // 30 sayfaya kadar döngü oluştur
                String baseUrl = "https://www.firat.edu.tr/tr/page/news?page=" + page;
                Document mainPage = Jsoup.connect(baseUrl).get();
                Elements posts = mainPage.select(".item-content  a");
                //System.out.println("Posts verisi:" + posts);
                for (int i = 0; i < posts.size(); i += 2) {
                    Element post = posts.get(i);
                    String postUrl = post.absUrl("href");
                    if (!postUrl.isEmpty()) {
                        System.out.println("Duyuru URL'si: " + postUrl);
                        Document postPage = Jsoup.connect(postUrl).get();
                        Elements content = postPage.select(".post-content");

                        for (Element element : content) {
                            writer.write(element.text());
                            writer.newLine();
                        }

                        writer.newLine();
                    }
                }
            }

            writer.close();
            System.out.println("Tüm içerikler başarıyla duyuru.txt dosyasına yazıldı.");
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
