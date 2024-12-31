package io.github.federate.mybatis;

import org.xml.sax.*;
import org.xml.sax.helpers.DefaultHandler;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.File;
import java.util.Stack;

public class MyBatisXMLParser extends DefaultHandler {
    private ObjectMapper mapper;
    private ObjectNode result;
    private ArrayNode sqlMappings;
    private ObjectNode sqlFragments;
    private StringBuilder currentSql;
    private Stack<String> elementStack;
    private String currentId;

    public MyBatisXMLParser() {
        mapper = new ObjectMapper();
        result = mapper.createObjectNode();
        sqlMappings = result.putArray("sqlMappings");
        sqlFragments = result.putObject("sqlFragments");
        currentSql = new StringBuilder();
        elementStack = new Stack<>();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Please provide the XML file path as an argument.");
            System.exit(1);
        }

        String xmlPath = args[0];

        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setValidating(false);
        factory.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
        factory.setFeature("http://xml.org/sax/features/validation", false);

        SAXParser saxParser = factory.newSAXParser();
        MyBatisXMLParser handler = new MyBatisXMLParser();
        saxParser.parse(new File(xmlPath), handler);

        System.out.println(handler.getResult());
    }

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) {
        elementStack.push(qName);
        if (qName.equals("select") || qName.equals("insert") || qName.equals("update") || qName.equals("delete") || qName.equals("sql")) {
            currentId = attributes.getValue("id");
            currentSql.setLength(0);
        }
    }

    @Override
    public void endElement(String uri, String localName, String qName) {
        if (qName.equals("select") || qName.equals("insert") || qName.equals("update") || qName.equals("delete")) {
            ObjectNode stmtNode = sqlMappings.addObject();
            stmtNode.put("id", currentId);
            stmtNode.put("sqlCommandType", qName.toUpperCase());
            stmtNode.put("sql", currentSql.toString().trim());
        } else if (qName.equals("sql")) {
            sqlFragments.put(currentId, currentSql.toString().trim());
        }
        elementStack.pop();
    }

    @Override
    public void characters(char[] ch, int start, int length) {
        String topElement = elementStack.peek();
        if (topElement.equals("select") || topElement.equals("insert") || topElement.equals("update") || topElement.equals("delete") || topElement.equals("sql")) {
            currentSql.append(new String(ch, start, length));
        }
    }

    public String getResult() throws Exception {
        return mapper.writeValueAsString(result);
    }
}